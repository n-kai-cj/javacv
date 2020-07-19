import java.util.concurrent.atomic.AtomicBoolean;

public class FileCapture {
    public static void main(String[] args) {

        final String fileName = "fullhd.mp4";
        final FileReadCaptureThread reader = new FileReadCaptureThread();
        if (!reader.initialize(fileName)) {
            System.err.println("initialize error");
            return;
        }
        final int width = reader.getWidth();
        final int height = reader.getHeight();
        final int sampleRate = reader.getSampleRate();
        final int sampleSizeInBits = 16;
        final int channel = reader.getChannel();

        System.out.println(String.format("video: %dx%d, audio: %d[Hz], %d, %d", width, height, sampleRate, sampleSizeInBits, channel));

        final AtomicBoolean loopFlag = new AtomicBoolean(true);
        Thread vidThread = new Thread((() -> {
            while (loopFlag.get()) {
                byte[] video = reader.getVideo();
                if (video == null) {
                    sleep(30);
                    continue;
                }
                OpenCVFX.imshow("JavaFX Capture", video, width, height);
                if (reader.getVideoSize() < 5) {
                    sleep(30);
                }
            }
            OpenCVFX.destroyAllWindows();
        }));
        Thread audThread = new Thread(() -> {
            AudioPlay audioPlay = new AudioPlay(sampleRate, sampleSizeInBits, channel);
            audioPlay.initialize();
            while (loopFlag.get()) {
                byte[] audio = reader.getAudio();
                if (audio == null) {
                    sleep(10);
                    continue;
                }
                audioPlay.play(audio);
            }
            audioPlay.uninitialize();
        });

        vidThread.start();
        audThread.start();
        reader.start();

        while (loopFlag.get()) {
            if (!reader.getLoopFlag()) {
                loopFlag.set(false);
                break;
            }
            sleep(1000);
        }

        reader.stop();
        try {
            audThread.join();
            vidThread.join();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

    }

    private static void sleep(long ms) {
        try {
            Thread.sleep(ms);
        } catch (InterruptedException ignored) {
        }
    }
}
