import os
import requests
import structlog
from typing import Optional

logger = structlog.get_logger(__name__)

class AlertManager:
    """
    Sistema de alertas para notificar sobre eventos críticos del sistema,
    como fallos en ingesta o degradaciones del modelo.
    """
    
    def __init__(self, webhook_url: Optional[str] = None):
        """
        Inicializa el manager de alertas.
        Si no se proporciona webhook_url, busca la variable de entorno WEBHOOK_URL.
        """
        self.webhook_url = webhook_url or os.environ.get("WEBHOOK_URL")

    def send_alert(self, title: str, message: str, level: str = "warning") -> bool:
        """
        Envía una alerta. Registra localmente en log y envía webhook si está configurado.
        
        Args:
            title: Título de la alerta.
            message: Mensaje detallado.
            level: Nivel de severidad ('info', 'warning', 'critical').
            
        Returns:
            bool: True si se procesó correctamente.
        """
        # 1. Loggear la alerta localmente siempre
        log_context = {"alert_title": title, "alert_level": level}
        if level == "critical":
            logger.critical(message, **log_context)
        elif level == "warning":
            logger.warning(message, **log_context)
        else:
            logger.info(message, **log_context)

        # 2. Enviar Webhook si existe
        if self.webhook_url:
            return self._send_webhook(title, message, level)
        
        return True

    def _send_webhook(self, title: str, message: str, level: str) -> bool:
        payload = {
            "text": f"[{level.upper()}] *{title}*\n{message}"
        }
        try:
            response = requests.post(self.webhook_url, json=payload, timeout=5)
            response.raise_for_status()
            logger.info("Webhook alert sent successfully.")
            return True
        except requests.exceptions.RequestException as e:
            logger.error("Failed to send webhook alert", error=str(e))
            return False
