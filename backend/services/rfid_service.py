import logging
import serial
from config import settings

logger = logging.getLogger("rfid_service")


class RfidService:
    def __init__(self):
        self.port = settings.SERIAL_PORT
        self.baudrate = settings.BAUD_RATE
        self.serial_connection = None
        self._connect_serial()

    def _connect_serial(self):
        try:
            if self.serial_connection and self.serial_connection.is_open:
                return
            logger.info(f"Opening serial connection on {self.port} at {self.baudrate} baud...")
            self.serial_connection = serial.Serial(
                port=self.port, baudrate=self.baudrate, timeout=1.0,
                xonxoff=False, rtscts=False, dsrdtr=False,
            )
            self.serial_connection.reset_input_buffer()
            self.serial_connection.reset_output_buffer()
            logger.info(f"Serial port {self.port} connected.")
        except serial.SerialException as e:
            logger.error(f"Serial connection to {self.port} failed: {str(e)}. Will retry on next scan.")
            self.serial_connection = None

    def read_rfid_uid(self) -> str:
        try:
            if not self.serial_connection or not self.serial_connection.is_open:
                self._connect_serial()
                if not self.serial_connection:
                    return None

            if self.serial_connection.in_waiting > 0:
                raw_line = self.serial_connection.readline()
                if not raw_line:
                    return None
                decoded = raw_line.decode("utf-8", errors="ignore").strip()
                if not decoded:
                    return None
                if "UID:" in decoded:
                    decoded = decoded.replace("UID:", "").strip()
                logger.info(f"Detected UID: {decoded}")
                return decoded.upper()
            return None
        except Exception as e:
            logger.error(f"Error reading serial buffer: {str(e)}")
            if self.serial_connection:
                try:
                    self.serial_connection.close()
                except Exception:
                    pass
                self.serial_connection = None
            return None

    def send_acknowledgement(self, success: bool):
        try:
            if not self.serial_connection or not self.serial_connection.is_open:
                return
            self.serial_connection.write(b'1' if success else b'0')
            self.serial_connection.flush()
        except Exception as e:
            logger.error(f"Failed to send ack to ESP32: {str(e)}")


rfid_service = RfidService()
