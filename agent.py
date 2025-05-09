import asyncio
import logging
import os

from dotenv import load_dotenv

# from livekit import agents
from livekit import api
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
)
from livekit.agents.events import RunContext
from livekit.agents.llm import function_tool
from livekit.agents.voice import Agent
from livekit.plugins import deepgram, openai, silero
from livekit.plugins.elevenlabs import tts
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.protocol import sip as proto_sip

# from livekit.protocol.sip import TransferSIPParticipantRequest

# Configuración básica
load_dotenv(dotenv_path=".env")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice-agent")


class Assistant(Agent):
    """Asistente de voz personalizado."""

    def __init__(self) -> None:
        super().__init__(
            instructions="""
            Eres un asistente amable y respondes preguntas en español de forma clara y concisa.
            Proporcionas información útil y mantienes un tono cordial en todo momento.
        """
        )
        # Datos específicos de la sesión
        self.room_name = None
        self.participant_identity = None
        self.session_data = {}

    def set_data(self, key, value):
        """Almacena datos en el contexto."""
        self.session_data[key] = value

    def get_data(self, key, default=None):
        """Recupera datos del contexto."""
        return self.session_data.get(key, default)

    @function_tool()
    async def transfer_call(self, ctx: RunContext):
        """Transfiere la llamada de una manera despectiva, donde indiques tu molestia al no querer ser atendido."""

        room = ctx.userdata.ctx.room
        identity = room.local_participant.identity
        transfer_number = f"sip:5652917934@127.0.0.1:49999"
        dept_name = "Agente"
        ctx.userdata.selected_department = dept_name

        await self._handle_transfer(identity, transfer_number, dept_name)
        return f"Transferring to {dept_name} department."


    async def _handle_transfer(
        self, identity: str, transfer_number: str, department: str
    ) -> None:
        """
        Handle the transfer process with department-specific messaging.

        Args:
            identity (str): The participant's identity
            transfer_number (str): The number to transfer to
            department (str): The name of the department
        """
        await self.session.generate_reply(
            user_input=f"En un momento sera transferido al deparamento de {department}. Por favor, no cuelgue la llamada."
        )
        await asyncio.sleep(6)
        await self.transfer_call(identity, transfer_number)

    async def transfer_call(self, participant_identity: str, transfer_to: str) -> None:
        """
        Transfer the SIP call to another number.

        Args:
            participant_identity (str): The identity of the participant.
            transfer_to (str): The phone number to transfer the call to.
        """
        logger.info(f"Transferring call for participant {participant_identity} to {transfer_to}")

        try:
            userdata = self.session.userdata
            if not userdata.livekit_api:
                livekit_url = os.getenv('LIVEKIT_URL')
                api_key = os.getenv('LIVEKIT_API_KEY')
                api_secret = os.getenv('LIVEKIT_API_SECRET')
                logger.debug(f"Initializing LiveKit API client with URL: {livekit_url}")
                userdata.livekit_api = api.LiveKitAPI(
                    url=livekit_url,
                    api_key=api_key,
                    api_secret=api_secret
                )

            transfer_request = proto_sip.TransferSIPParticipantRequest(
                participant_identity=participant_identity,
                room_name=userdata.ctx.room.name,
                transfer_to=transfer_to,
                play_dialtone=True
            )
            logger.debug(f"Transfer request: {transfer_request}")

            await userdata.livekit_api.sip.transfer_sip_participant(transfer_request)
            logger.info(f"Successfully transferred participant {participant_identity} to {transfer_to}")

        except Exception as e:
            logger.error(f"Failed to transfer call: {e}", exc_info=True)
            await self.session.generate_reply(user_input="Lo sentimos, no fue posible llevar a cabo la transferencia en estos momentos. Hay algo mas en lo que te pueda apoyar?")


def prewarm(proc: JobProcess):
    """Precarga el modelo VAD para detección de voz."""
    proc.userdata["vad"] = silero.VAD.load(activation_threshold=0.9)


async def entrypoint(ctx: JobContext):
    """Punto de entrada principal del asistente de voz."""

    # Conectar a la sala
    logger.info(f"Conectando a la sala {ctx.room.name}")
    await ctx.connect()

    # Esperar al primer participante
    participant = await ctx.wait_for_participant()
    logger.info(f"Iniciando asistente para {participant.identity}")

    # Crear instancia del asistente
    assistant = Assistant()
    assistant.room_name = ctx.room.name
    assistant.participant_identity = participant.identity

    # Configurar datos iniciales (ejemplo)
    assistant.set_data("start_time", "now")
    assistant.set_data("language", "es")

    # Configurar el motor TTS de ElevenLabs
    eleven_tts = tts.TTS(
        voice_id="YKUjKbMlejgvkOZlnnvt",
        api_key=os.environ.get("ELEVENLABS_API_KEY"),
        model="eleven_turbo_v2_5",
        # voice_settings=tts.Voice(
        #     id="YKUjKbMlejgvkOZlnnvt",
        #     name="Alejandro Ballesteros",
        #     category="professional",
        #     # voice_settings=tts.VoiceSettings(stability=1, similarity_boost=1, use_speaker_boost=True),
        # ),
        language="es",
        streaming_latency=3,
        enable_ssml_parsing=False,
        chunk_length_schedule=[80, 120, 200, 260],
    )

    # Crear y configurar la sesión
    session = AgentSession(
        stt=deepgram.STT(model="nova-2", language="es"),  # Configurado para español
        llm=openai.LLM(model="gpt-4o-mini", temperature=0.7),
        tts=eleven_tts,  # Usamos ElevenLabs en lugar de Cartesia
        vad=ctx.proc.userdata["vad"],
        turn_detection=MultilingualModel(),
        # turn_detector=turn_detector.EOUModel(),
    )

    # Iniciar la sesión
    await session.start(
        room=ctx.room,
        agent=assistant,
        # room_input_options=RoomInputOptions(
        #     # LiveKit Cloud enhanced noise cancellation
        #     noise_cancellation=noise_cancellation.BVC(),
        # ),
    )

    # Mensaje de bienvenida
    await session.generate_reply(
        instructions="Saluda al usuario en español y ofrece tu ayuda."
    )


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            # agents.cli.run_app(WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        )
    )
