import com.fazecast.jSerialComm.SerialPort;

public class ArduinoSerialWriter {
    private SerialPort serialPort = null;

    public void setupSerialComm(){
        // Find and open the serial port
        SerialPort[] ports = SerialPort.getCommPorts();
        for (SerialPort port : ports) {
            if (port.getSystemPortName().contains(Consts.PORT_NAME)) {
                System.out.println("Found correct port!");
                this.serialPort = port;
                break;
            }
        }
        if (this.serialPort == null) {
            System.out.println("WARNING: Arduino port not found.");
            return;
        }
        if (!this.serialPort.openPort()) {
            System.out.println("ERROR: Failed to open Arduino port.");
            return;
        }
    }

    public void closeSerialComm(){
        turnOnCoils(Consts.RESET_COILS);
        this.serialPort.closePort();
    }

    public void turnOnCoils(String coilString){
        this.serialPort.writeBytes(coilString.getBytes(), coilString.length());
    }

    public boolean isArduinoConnected() {
        return (this.serialPort == null) ? false : true;
    }
}