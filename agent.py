import logging
import os

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
)
from livekit.plugins import deepgram, noise_cancellation, openai, silero, turn_detector
from livekit.plugins.elevenlabs import tts
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# Configuración básica
load_dotenv(dotenv_path=".env")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice-agent")

class Assistant(Agent):
    """Asistente de voz personalizado."""

    def __init__(self) -> None:
        super().__init__(instructions="""
            Eres un asistente amable y respondes preguntas en español de forma clara y concisa.
            Proporcionas información útil y mantienes un tono cordial en todo momento.
        """)
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
    agents.cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        prewarm_fnc=prewarm,
    ))