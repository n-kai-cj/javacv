import org.bytedeco.ffmpeg.avcodec.AVCodec;
import org.bytedeco.ffmpeg.avcodec.AVCodecContext;
import org.bytedeco.ffmpeg.avcodec.AVPacket;
import org.bytedeco.ffmpeg.avformat.AVFormatContext;
import org.bytedeco.ffmpeg.avformat.AVStream;
import org.bytedeco.ffmpeg.avutil.AVDictionary;
import org.bytedeco.ffmpeg.avutil.AVFrame;
import org.bytedeco.ffmpeg.global.avcodec;
import org.bytedeco.ffmpeg.global.avformat;
import org.bytedeco.ffmpeg.global.avutil;
import org.bytedeco.ffmpeg.global.swresample;
import org.bytedeco.ffmpeg.swresample.SwrContext;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.PointerPointer;

import java.nio.ByteBuffer;
import java.util.ArrayList;

public class FileReadCaptureThread implements Runnable {
    private boolean loopFlag = false;
    private AVFormatContext avFormatContext = null;
    private AVPacket avPacket = null;
    private Thread thread = null;

    // video
    private AVCodecContext videoDecodeContext = null;
    private AVFrame avFrameYuv = null;
    private int videoStreamIndex;
    private final ArrayList<byte[]> videoList = new ArrayList<>();
    private FFmpegPixelFormatConverter pixelConverter = null;
    private ByteBuffer videoBuffer = ByteBuffer.allocate(1);
    private BytePointer videoBytePointer = new BytePointer(1);
    private int width = -1;
    private int height = -1;
    private int fps = -1;
    private final static int VIDEO_MAX_BUFFER_SIZE = 10;

    // audio
    private AVCodecContext audioDecodeContext = null;
    private AVFrame avFramePcm = null;
    private int audioStreamIndex;
    private final ArrayList<byte[]> audioList = new ArrayList<>();
    private SwrContext swr = null;
    private BytePointer[] samples_out = null;
    private final PointerPointer<Pointer> samples_out_ptr = new PointerPointer<>(AVFrame.AV_NUM_DATA_POINTERS);
    private int s16BufferSize = -1;
    private int sampleRate = -1;
    private int channel = -1;
    private final static int AUDIO_MAX_BUFFER_SIZE = 10;

    static {
        avformat.avformat_network_init();
    }

    public FileReadCaptureThread() {
    }

    public void start() {
        this.loopFlag = true;
        this.thread = new Thread(this);
        this.thread.start();
    }

    public void stop() {
        this.loopFlag = false;
        try {
            this.thread.interrupt();
            this.thread.join();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        this.thread = null;
        uninitialize();
    }

    public byte[] getVideo() {
        if (this.videoList.size() == 0) {
            return null;
        }
        synchronized (this.videoList) {
            return this.videoList.remove(0);
        }
    }

    public byte[] getAudio() {
        if (this.audioList.size() == 0) {
            return null;
        }
        synchronized (this.audioList) {
            return this.audioList.remove(0);
        }
    }

    public boolean initialize(String fileName) {
        int ret;
        this.avFormatContext = avformat.avformat_alloc_context();
        ret = avformat.avformat_open_input(this.avFormatContext, fileName, null, null);
        if (ret < 0) {
            System.err.println("avformat_open_input error: " + av_err2str(ret));
            return false;
        }
        ret = avformat.avformat_find_stream_info(this.avFormatContext, (AVDictionary) null);
        if (ret < 0) {
            System.err.println("avformat_find_stream_info error: " + av_err2str(ret));
            return false;
        }
        this.videoStreamIndex = avformat.av_find_best_stream(this.avFormatContext, avutil.AVMEDIA_TYPE_VIDEO, -1, -1, (AVCodec) null, 9);
        if (this.videoStreamIndex < 0) {
            System.err.println("video stream invalid " + this.videoStreamIndex);
            return false;
        }
        this.audioStreamIndex = avformat.av_find_best_stream(this.avFormatContext, avutil.AVMEDIA_TYPE_AUDIO, -1, -1, (AVCodec) null, 9);
        if (this.audioStreamIndex < 0) {
            System.err.println("audio stream invalid " + this.audioStreamIndex);
            return false;
        }

        // video decoder initialize
        AVStream videoStream = this.avFormatContext.streams(this.videoStreamIndex);
        System.out.println("video codec = " + avcodec_get_name(videoStream.codecpar().codec_id()));
        AVCodec videoCodec = avcodec.avcodec_find_decoder(videoStream.codecpar().codec_id());
        if (videoCodec == null) {
            System.err.println("find video decoder error");
            return false;
        }
        this.videoDecodeContext = avcodec.avcodec_alloc_context3(videoCodec);
        ret = avcodec.avcodec_parameters_to_context(this.videoDecodeContext, videoStream.codecpar());
        if (ret < 0) {
            System.err.println("video codec_parameters_to_context error: " + av_err2str(ret));
            return false;
        }
        ret = avcodec.avcodec_open2(this.videoDecodeContext, videoCodec, (AVDictionary) null);
        if (ret < 0) {
            System.err.println("video codec open error: " + av_err2str(ret));
            return false;
        }
        this.avFrameYuv = avutil.av_frame_alloc();
        if (this.avFrameYuv == null) {
            return false;
        }
        this.width = this.videoDecodeContext.width();
        this.height = this.videoDecodeContext.height();
        this.fps = videoStream.avg_frame_rate().num();

        // audio decoder initialize
        AVStream audioStream = this.avFormatContext.streams(this.audioStreamIndex);
        System.out.println("audio codec = " + avcodec_get_name(audioStream.codecpar().codec_id()));
        AVCodec audioCodec = avcodec.avcodec_find_decoder(audioStream.codecpar().codec_id());
        if (audioCodec == null) {
            System.err.println("find audio codec error");
            return false;
        }
        this.audioDecodeContext = avcodec.avcodec_alloc_context3(audioCodec);
        ret = avcodec.avcodec_parameters_to_context(this.audioDecodeContext, audioStream.codecpar());
        if (ret < 0) {
            System.err.println("audio codec_parameters_to_context error: " + av_err2str(ret));
            return false;
        }
        ret = avcodec.avcodec_open2(this.audioDecodeContext, audioCodec, (AVDictionary) null);
        if (ret < 0) {
            System.err.println("audio codec open error: " + av_err2str(ret));
            return false;
        }
        this.avFramePcm = avutil.av_frame_alloc();
        if (this.avFramePcm == null) {
            return false;
        }
        this.s16BufferSize = avutil.av_samples_get_buffer_size((IntPointer) null,
                this.audioDecodeContext.channels(), this.audioDecodeContext.frame_size(),
                avutil.AV_SAMPLE_FMT_S16, 1);
        this.swr = swresample.swr_alloc_set_opts(null,
                this.audioDecodeContext.channel_layout(), avutil.AV_SAMPLE_FMT_S16, this.audioDecodeContext.sample_rate(),
                this.audioDecodeContext.channel_layout(), avutil.AV_SAMPLE_FMT_FLTP, this.audioDecodeContext.sample_rate(),
                0, null);
        if (this.swr == null) {
            System.err.println("audio swresample alloc error");
            return false;
        }
        ret = swresample.swr_init(this.swr);
        if (ret < 0) {
            System.err.println("audio swresample init error: " + av_err2str(ret));
            return false;
        }
        int planes = avutil.av_sample_fmt_is_planar(this.audioDecodeContext.sample_fmt()) != 0 ? this.audioDecodeContext.channels() : 1;
        int data_size = avutil.av_samples_get_buffer_size((IntPointer) null,
                this.audioDecodeContext.channels(), this.audioDecodeContext.frame_size(), avutil.AV_SAMPLE_FMT_S16, 1) / planes;
        this.samples_out = new BytePointer[planes];
        for (int i = 0; i < this.samples_out.length; i++) {
            this.samples_out[i] = new BytePointer(avutil.av_malloc(data_size)).capacity(data_size);
        }
        for (int i = 0; i < this.samples_out.length; i++) {
            this.samples_out_ptr.put(i, this.samples_out[i]);
        }
        this.sampleRate = this.audioDecodeContext.sample_rate();
        this.channel = this.audioDecodeContext.channels();

        this.avPacket = new AVPacket();
        avcodec.av_init_packet(this.avPacket);
        this.avPacket.data(null);
        this.avPacket.size(0);

        System.out.println("initialize succeed");

        return true;
    }

    public void uninitialize() {
        if (this.videoDecodeContext != null) {
            avcodec.avcodec_close(this.videoDecodeContext);
            avcodec.avcodec_free_context(this.videoDecodeContext);
            this.videoDecodeContext = null;
        }
        if (this.audioDecodeContext != null) {
            avcodec.avcodec_close(this.audioDecodeContext);
            avcodec.avcodec_free_context(this.audioDecodeContext);
            this.audioDecodeContext = null;
        }
        if (this.avFrameYuv != null) {
            avutil.av_frame_free(this.avFrameYuv);
            this.avFrameYuv = null;
        }
        if (this.avFramePcm != null) {
            avutil.av_frame_free(this.avFramePcm);
            this.avFramePcm = null;
        }
        if (this.avPacket != null) {
            avcodec.av_packet_unref(this.avPacket);
            this.avPacket = null;
        }
        if (this.avFormatContext != null) {
            avformat.avformat_close_input(this.avFormatContext);
            avformat.avformat_free_context(this.avFormatContext);
            this.avFormatContext = null;
        }
        if (this.pixelConverter != null) {
            this.pixelConverter.uninitialize();
            this.pixelConverter = null;
        }
    }

    @Override
    public void run() {
        System.out.println("loop start");
        while (this.loopFlag) {
            if (avformat.av_read_frame(this.avFormatContext, this.avPacket) < 0) {
                this.loopFlag = false;
                break;
            }
            if (this.avPacket.stream_index() == this.videoStreamIndex) {
                videoDecode();
            } else if (this.avPacket.stream_index() == this.audioStreamIndex) {
                audioDecode();
            } else {
                System.err.println("stream index is invalid " + this.avPacket.stream_index());
            }
            avcodec.av_packet_unref(this.avPacket);
            if (this.audioList.size() > AUDIO_MAX_BUFFER_SIZE / 2 || this.videoList.size() > VIDEO_MAX_BUFFER_SIZE / 2) {
                sleep(1000 / this.fps);
            }
        }
        System.out.println("loop exit");
        uninitialize();
    }

    private void audioDecode() {
        int ret;
        ret = avcodec.avcodec_send_packet(this.audioDecodeContext, this.avPacket);
        if (ret < 0) {
            System.err.println("failed: audio avcodec_send_apcket = " + av_err2str(ret));
            return;
        }
        int avPktSize = this.avPacket.size();
        avcodec.av_packet_unref(this.avPacket);
        ret = avcodec.avcodec_receive_frame(this.audioDecodeContext, this.avFramePcm);
        if (ret < 0) {
            System.err.println("failed: audio gotPacket is null. avPacket.size=+" + avPktSize + ": " + av_err2str(ret));
            return;
        }
        int nb_samples = this.avFramePcm.nb_samples();

        int dst_nb_samples = (int) avutil.av_rescale_rnd(
                swresample.swr_get_delay(this.swr, this.audioDecodeContext.sample_rate()) + nb_samples,
                this.audioDecodeContext.sample_rate(), this.audioDecodeContext.sample_rate(), avutil.AV_ROUND_UP);
        ret = swresample.swr_convert(this.swr, this.samples_out_ptr, dst_nb_samples,
                this.avFramePcm.extended_data(), nb_samples);
        if (ret < 0) {
            System.err.println("audio swresample convert error: " + av_err2str(ret));
            return;
        }
        byte[] s16 = new byte[this.s16BufferSize];
        this.samples_out[0].get(s16, 0, s16.length);
        synchronized (this.audioList) {
            this.audioList.add(s16);
            if (this.audioList.size() > AUDIO_MAX_BUFFER_SIZE) {
                System.err.println(String.format("audio buffer overflow %d", this.audioList.size()));
                this.audioList.remove(0);
            }
        }
    }

    private void videoDecode() {
        int ret;
        ret = avcodec.avcodec_send_packet(this.videoDecodeContext, this.avPacket);
        if (ret < 0) {
            System.err.println("failed: video avcodec_send_packet = " + av_err2str(ret));
            return;
        }
        int avPktSize = this.avPacket.size();
        avcodec.av_packet_unref(this.avPacket);
        ret = avcodec.avcodec_receive_frame(this.videoDecodeContext, this.avFrameYuv);
        if (ret < 0) {
            System.err.println("failed: video gotPacket is null. avPacket.size=+" + avPktSize + ": " + av_err2str(ret));
            return;
        }
        int width = this.avFrameYuv.width();
        int height = this.avFrameYuv.height();
        if (this.pixelConverter == null) {
            this.pixelConverter = new FFmpegPixelFormatConverter();
            if (!this.pixelConverter.initialize(width, height, this.avFrameYuv.format(), avutil.AV_PIX_FMT_RGB24, 16)) {
                System.err.println("video pixel converter initialize error");
                return;
            }
        }
        int yuvLen = avutil.av_image_get_buffer_size(this.avFrameYuv.format(), width, height, 1);
        byte[] yuvByte = new byte[yuvLen];
        this.videoBuffer.clear();
        if (this.videoBuffer.remaining() != yuvLen) {
            this.videoBuffer = ByteBuffer.allocateDirect(yuvLen);
            this.videoBytePointer = new BytePointer(this.videoBuffer);
        }
        ret = avutil.av_image_copy_to_buffer(this.videoBytePointer, yuvLen,
                this.avFrameYuv.data(), this.avFrameYuv.linesize(),
                this.avFrameYuv.format(), width, height, 1);
        if (ret < 0) {
            System.err.println("video decode image copy error: " + av_err2str(ret));
            return;
        }
        this.videoBuffer.get(yuvByte);
        int rgbLen = avutil.av_image_get_buffer_size(avutil.AV_PIX_FMT_RGB24, width, height, 1);
        byte[] rgb = new byte[rgbLen];
        rgb = this.pixelConverter.convert(width, height, yuvByte, rgb);
        synchronized (this.videoList) {
            this.videoList.add(rgb);
            if (this.videoList.size() > VIDEO_MAX_BUFFER_SIZE) {
                System.err.println(String.format("video buffer overflow %d ", this.videoList.size()));
                this.videoList.remove(0);
            }
        }
    }

    public boolean getLoopFlag() {
        return this.loopFlag;
    }

    public int getWidth() {
        return this.width;
    }

    public int getHeight() {
        return this.height;
    }

    public int getVideoSize() {
        return this.videoList.size();
    }

    public int getSampleRate() {
        return this.sampleRate;
    }

    public int getChannel() {
        return this.channel;
    }

    private void sleep(long ms) {
        try {
            Thread.sleep(ms);
        } catch (InterruptedException ignored) {
        }
    }

    private static String av_err2str(int errnum) {
        byte[] errbuf = new byte[avutil.AV_ERROR_MAX_STRING_SIZE];
        avutil.av_make_error_string(errbuf, avutil.AV_ERROR_MAX_STRING_SIZE, errnum);
        return new String(errbuf);
    }

    private static String avcodec_get_name(int id) {
        return avcodec.avcodec_get_name(id).getString();
    }


}
