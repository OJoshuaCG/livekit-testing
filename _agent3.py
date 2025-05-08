#!/usr/bin/env python
"""
Voice Assistant Agent usando LiveKit

Este módulo implementa un asistente de voz utilizando LiveKit y varios plugins
para proporcionar una experiencia conversacional completa.
"""

import logging
import os
from typing import Optional

from dotenv import load_dotenv
from functions import AssistantFnc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import deepgram, openai, silero, turn_detector
from livekit.plugins.elevenlabs import tts

# Configuración del logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("voice-agent")


class VoiceAssistantApp:
    """
    Clase principal para gestionar la aplicación del asistente de voz.
    Encapsula la configuración y funcionalidad principal.
    """

    # Configuración del sistema para el asistente
    SYSTEM_PROMPT = """
    Eres un asistente amable y responde preguntas de forma clara y concisa.
    Proporciona información útil y mantén un tono cordial en todo momento.
    """

    def __init__(self):
        """Inicializa la aplicación cargando variables de entorno."""
        # Cargar variables de entorno
        self._load_environment()

    @staticmethod
    def _load_environment():
        """Carga las variables de entorno desde el archivo .env.local"""
        env_path = ".env.local"
        if os.path.exists(env_path):
            load_dotenv(dotenv_path=env_path)
        else:
            logger.warning(f"Archivo {env_path} no encontrado. Usando variables de entorno del sistema.")

    @staticmethod
    def prewarm(proc: JobProcess):
        """
        Función de precalentamiento que carga modelos pesados antes de procesar trabajos.

        Args:
            proc: El proceso de trabajo donde almacenar componentes precargados.
        """
        proc.userdata["vad"] = silero.VAD.load(activation_threshold=0.9)

    def _create_tts_engine(self) -> tts.TTS:
        """
        Crea y configura el motor de text-to-speech.

        Returns:
            Una instancia configurada del motor TTS de ElevenLabs.
        """
        api_key = os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            logger.error("No se encontró ELEVENLABS_API_KEY en las variables de entorno")
            raise ValueError("API key de ElevenLabs es requerida")

        return tts.TTS(
            api_key=api_key,
            model="eleven_turbo_v2_5",
            voice=tts.Voice(
                id="YKUjKbMlejgvkOZlnnvt",
                name="Alejandro Ballesteros",
                category="professional",
                settings=tts.VoiceSettings(
                    stability=1,
                    similarity_boost=1,
                    use_speaker_boost=True
                ),
            ),
            language="es",
            streaming_latency=3,
            enable_ssml_parsing=False,
            chunk_length_schedule=[80, 120, 200, 260],
        )

    def _create_agent(self, vad, fnc_ctx) -> VoicePipelineAgent:
        """
        Crea y configura el agente de voz con todos sus componentes.

        Args:
            vad: Detector de actividad de voz precargado.
            fnc_ctx: Contexto de funciones del asistente.

        Returns:
            Una instancia configurada del agente de voz.
        """
        # Inicializar el contexto de chat
        initial_ctx = llm.ChatContext().append(
            role="system",
            text=self.SYSTEM_PROMPT,
        )

        # Crear el agente
        return VoicePipelineAgent(
            vad=vad,
            stt=deepgram.STT(language="es"),
            llm=openai.LLM(model="gpt-4o-mini", temperature=0.7),
            tts=self._create_tts_engine(),
            turn_detector=turn_detector.EOUModel(),
            min_endpointing_delay=0.5,
            max_endpointing_delay=5.0,
            chat_ctx=initial_ctx,
            fnc_ctx=fnc_ctx,
            max_nested_fnc_calls=5,
        )

    async def entrypoint(self, ctx: JobContext):
        """
        Punto de entrada principal para la aplicación.

        Args:
            ctx: El contexto del trabajo actual.
        """
        logger.info(f"Conectando a la sala {ctx.room.name}")
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

        # Esperar a que se conecte el primer participante
        participant = await ctx.wait_for_participant()
        logger.info(f"Iniciando asistente de voz para el participante {participant.identity}")

        # Configurar el contexto de funciones
        fnc_ctx = AssistantFnc()
        fnc_ctx.room_name = ctx.room.name
        fnc_ctx.participant_identity = participant.identity

        # Crear y arrancar el agente
        agent = self._create_agent(ctx.proc.userdata["vad"], fnc_ctx)
        agent.start(ctx.room, participant)

        # Mensaje de bienvenida
        await agent.say(
            "¡Hola! Bienvenido estoy aquí para ayudarte en lo que necesites.",
            allow_interruptions=True
        )

    def run(self):
        """Inicia la aplicación utilizando el CLI de LiveKit."""
        cli.run_app(
            WorkerOptions(
                entrypoint_fnc=self.entrypoint,
                prewarm_fnc=self.prewarm,
            ),
        )


if __name__ == "__main__":
    app = VoiceAssistantApp()
    app.run()