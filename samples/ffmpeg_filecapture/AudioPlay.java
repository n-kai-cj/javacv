import javax.sound.sampled.*;
import java.nio.ByteOrder;

public class AudioPlay {
    private final int sampleRate;
    private final int sampleSizeInBits;
    private final int channel;
    private final static Object lock = new Object();
    private static SourceDataLine sourceDataLine = null;

    public AudioPlay(int rate, int ssize, int c) {
        this.sampleRate = rate;
        this.sampleSizeInBits = ssize;
        this.channel = c;
    }

    public void initialize() {
        synchronized (lock) {
            if (sourceDataLine == null) {
                AudioFormat format = new AudioFormat(this.sampleRate, this.sampleSizeInBits, this.channel, true, Boolean.parseBoolean(ByteOrder.nativeOrder().toString()));
                DataLine.Info info = new DataLine.Info(SourceDataLine.class, format);
                try {
                    sourceDataLine = (SourceDataLine) AudioSystem.getLine(info);
                    sourceDataLine.open(format);
                    sourceDataLine.start();
                } catch (LineUnavailableException e1) {
                    e1.printStackTrace();
                }
            }
        }
    }

    public void uninitialize() {
        synchronized (lock) {
            if (sourceDataLine != null) {
                sourceDataLine.stop();
                sourceDataLine.close();
                sourceDataLine = null;
            }
        }
    }

    public void play(byte[] audio) {
        synchronized (lock) {
            if (sourceDataLine != null && sourceDataLine.isOpen()) {
                sourceDataLine.write(audio, 0, audio.length);
            }
        }
    }

}
