import org.joml.Matrix4f;
import org.joml.Vector3f;
import org.lwjgl.BufferUtils;
import org.lwjgl.glfw.GLFWErrorCallback;
import org.lwjgl.glfw.GLFWVidMode;
import org.lwjgl.opengl.GL;
import org.lwjgl.system.MemoryStack;

import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicLong;

import static org.lwjgl.glfw.Callbacks.glfwFreeCallbacks;
import static org.lwjgl.glfw.GLFW.*;
import static org.lwjgl.opengl.GL11.*;

public class OpenGlViewer implements Runnable {
    private int width;
    private int height;
    private final String title;
    private float zoom;
    private int mouseX, mouseY;
    private final Vector3f center = new Vector3f();
    private final Vector3f translate = new Vector3f();
    private static final int GRID_SiZE = 20;
    private static final float AXIS_SIZE = 1.0f;
    private static final float FOV = 90.0f;
    private static final float Z_NEAR = 0.001f;
    private static final float Z_FAR = 100.0f;
    private boolean isGrid = true;
    private boolean isAxis = true;
    private float pitch, yaw, roll;
    private long window = 0;
    private boolean isLaunch = false;
    private boolean isRunning = false;
    private final ArrayList<Matrix4f> cams = new ArrayList<>();
    private int maxCamNum = 20;

    public OpenGlViewer(int width, int height, String title) {
        this.width = width;
        this.height = height;
        this.title = title;
    }

    public void launch() {
        isLaunch = true;
        new Thread(this).start();
    }

    public void destroy() {
        if (this.isRunning)
            exit(this.window);
    }

    public boolean isRunning() {
        return isRunning;
    }

    public boolean isLaunch() {
        return isLaunch;
    }

    public void showGrid(boolean isShow) {
        isGrid = isShow;
    }

    public void showAxis(boolean isShow) {
        isAxis = isShow;
    }

    public void addCameraPose(float[] f) {
        synchronized (cams) {
            cams.add(new Matrix4f().set(f));
            if (maxCamNum > 0 && cams.size() > maxCamNum) {
                cams.remove(0);
            }
        }
    }

    public void setMaxCamNum(int num) {
        maxCamNum = num;
    }

    @Override
    public void run() {
        isLaunch = true;
        AtomicLong window = new AtomicLong();
        if (!initialize(window, title)) {
            System.err.println("init failed");
            return;
        }
        this.window = window.get();
        resetView();
        cams.clear();
        Matrix4f mat = new Matrix4f();
        FloatBuffer fb = BufferUtils.createFloatBuffer(16);

        // Run the rendering loop until the user has attempted to close
        // the window or has pressed the ESCAPE key.
        while (isRunning = !glfwWindowShouldClose(window.get())) {
            glViewport(0, 0, width, height);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the framebuffer
            // draw axis
            if (isAxis) drawAxis();

            // update camera pose
            updateCamera();

            glMatrixMode(GL_PROJECTION);
            glLoadMatrixf(mat.setPerspective((float) Math.toRadians(FOV), (float) width / height, Z_NEAR, Z_FAR).get(fb));
            glMatrixMode(GL_MODELVIEW);
            // Load camera view matrix into 'mat':
            glLoadMatrixf(mat
                    .translation(0, 0, -zoom)
                    .translate(translate.x, -translate.y, 0)
                    .rotateZ(roll)
                    .rotateX(pitch)
                    .rotateY(yaw)
                    .translate(-center.x, -center.y, -center.z)
                    .get(fb));

            // draw grid
            if (isGrid) drawGrid();

            // apply model transformation to 'mat':
            glLoadMatrixf(mat
                    .translate(center)
                    .get(fb));

            glfwSwapBuffers(window.get()); // swap the color buffers
            // Poll for window events. The key callback above will only be
            // invoked during this call.
            glfwPollEvents();
        }
        uninitialize(window.get());
        isLaunch = false;
    }

    private void updateCamera() {
        if (cams.size() <= 0) {
            return;
        }
        synchronized (cams) {
            float w = 0.3f;
            drawCamera(cams.get(0), w);
            for (int i = 0; i < cams.size() - 1; ++i) {
                Matrix4f m1 = cams.get(i);
                Matrix4f m2 = cams.get(i + 1);
                drawCamera(m2, w);
                drawTrajectory(m1, m2);
            }
        }
    }

    // ----------------------------------------------------------------------------------------------------
    // Initialize / Uninitialize
    // ----------------------------------------------------------------------------------------------------
    private boolean initialize(AtomicLong handle, String title) {
        // Setup an error callback. The default implementation
        // will print the error message in System.err.
        GLFWErrorCallback.createPrint(System.err).set();

        // Initialize GLFW. Most GLFW functions will not work before doing this.
        if (!glfwInit()) {
            System.err.println("Unable to initialize GLFW");
            return false;
        }

        // Configure GLFW
        glfwDefaultWindowHints(); // optional, the current window hints are already the default
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE); // the window will stay hidden after creation
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE); // the window will be resizable

        // Create the window
        long window = glfwCreateWindow(width, height, title, org.lwjgl.system.MemoryUtil.NULL, org.lwjgl.system.MemoryUtil.NULL);
        if (window == org.lwjgl.system.MemoryUtil.NULL) {
            System.err.println("Failed to create the GLFW window");
            return false;
        }

        // Setup resize callback
        glfwSetFramebufferSizeCallback(window, (windowHandle, width, height) -> {
            if (width > 0 && height > 0) {
                this.width = width;
                this.height = height;
            }
        });
        // Setup a key callback
        glfwSetKeyCallback(window, this::controlByKey);
        // Setup a mouse cursor and button callback
        glfwSetCursorPosCallback(window, (windowHandle, xpos, ypos) -> mouseInput(window, xpos, ypos));
        // Setup a mouse scroll callback
        glfwSetScrollCallback(window, (windowHandle, x, y) -> scrollInput(y));

        // Get the thread stack and push a new frame
        try (MemoryStack stack = MemoryStack.stackPush()) {
            IntBuffer pWidth = stack.mallocInt(1); // int*
            IntBuffer pHeight = stack.mallocInt(1); // int*

            // Get the window size passed to glfwCreateWindow
            glfwGetWindowSize(window, pWidth, pHeight);

            // Get the resolution of the primary monitor
            GLFWVidMode vidmode = glfwGetVideoMode(glfwGetPrimaryMonitor());
            if (vidmode == null) {
                System.err.println("Failed to get video mode");
                return false;
            }

            // Center the window
            glfwSetWindowPos(
                    window,
                    (vidmode.width() - pWidth.get(0)) / 2,
                    (vidmode.height() - pHeight.get(0)) / 2
            );

        } // the stack frame is popped automatically

        // Make the OpenGL context current
        glfwMakeContextCurrent(window);
        // Enable v-sync
        glfwSwapInterval(1);

        // Make the window visible
        glfwShowWindow(window);

        // This line is critical for LWJGL's interoperation with GLFW's
        // OpenGL context, or any context that is managed externally.
        // LWJGL detects the context that is current in the current thread,
        // creates the GLCapabilities instance and makes the OpenGL
        // bindings available for use.
        GL.createCapabilities();

        // Set the clear color
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnable(GL_DEPTH_TEST);

        handle.set(window);

        return true;
    }

    private void uninitialize(long window) {
        // Free the window callbacks and destroy the window
        glfwFreeCallbacks(window);
        glfwDestroyWindow(window);

        // Terminate GLFW and free the error callback
        glfwTerminate();
        Objects.requireNonNull(glfwSetErrorCallback(null)).free();
    }

    private void exit(long window) {
        glfwSetWindowShouldClose(window, true);
    }

    // ----------------------------------------------------------------------------------------------------
    // OpenGL View Control
    // ----------------------------------------------------------------------------------------------------
    private void controlByKey(long window, int key, int scancode, int action, int mods) {
        // ESC to exit
        if (key == GLFW_KEY_ESCAPE && action == GLFW_RELEASE) {
            exit(window);
        }
        // Backspace to reset view
        if (key == GLFW_KEY_BACKSPACE && action == GLFW_RELEASE) {
            resetView();
        }
    }

    private void mouseInput(long window, double xpos, double ypos) {
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_1) == GLFW_PRESS && glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_2) == GLFW_PRESS) {
            // rotate around z-axis when Left+Right pressed
            roll += (xpos - mouseX) * 0.01f;
        } else if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_1) == GLFW_PRESS) {
            // translate when Left pressed
            translate.x += (xpos - mouseX) * 0.01f;
            translate.y += (ypos - mouseY) * 0.01f;
        } else if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_2) == GLFW_PRESS) {
            // rotate around x-axis and y-axis when Right click
            yaw += ((int) xpos - mouseX) * 0.01f;
            pitch += ((int) ypos - mouseY) * 0.01f;
        }
        mouseX = (int) xpos;
        mouseY = (int) ypos;
    }

    private void scrollInput(double y) {
        double inc = Math.abs(zoom * 0.1);
        if (inc < 0.1) inc = 0.1;
        zoom = (float) (y > 0 ? zoom - inc : zoom + inc);
    }

    private void resetView() {
        zoom = 20;
        center.set(0, 0, 0);
        translate.set(0, 0, 0);
        pitch = 0.3f;
        yaw = 0.2f;
        roll = 0.0f;
    }

    // ----------------------------------------------------------------------------------------------------
    // Draw on OpenGL
    // ----------------------------------------------------------------------------------------------------
    private void drawTrajectory(Matrix4f cam1, Matrix4f cam2) {
        glPushMatrix();
        glColor3f(0.8f, 0.2f, 0.2f);
        glLineWidth(1.0f);
        drawLine(cam1.m30(), cam1.m31(), cam1.m32(), cam2.m30(), cam2.m31(), cam2.m32());
        glPopMatrix();
    }

    private void drawCamera(Matrix4f cam_pose, float width) {
        glPushMatrix();
        float[] f = new float[16];
        cam_pose.get(f);
        glMultMatrixf(f);
        drawFrustum(width);
        glPopMatrix();
    }

    private void drawFrustum(float w) {
        float h = w * 0.75f;
        float z = w * 1.25f;

        glColor3f(0.9f, 0.7f, 0.7f);
        drawRectangle(w, h, z);

        glColor3f(0.8f, 0.2f, 0.2f);
        glLineWidth(1.0f);
        drawLine(0.0f, 0.0f, 0.0f, w, h, z);
        drawLine(0.0f, 0.0f, 0.0f, w, -h, z);
        drawLine(0.0f, 0.0f, 0.0f, -w, -h, z);
        drawLine(0.0f, 0.0f, 0.0f, -w, h, z);
        drawLine(w, h, z, w, -h, z);
        drawLine(-w, h, z, -w, -h, z);
        drawLine(-w, h, z, w, h, z);
        drawLine(-w, -h, z, w, -h, z);
    }

    private void drawRectangle(float x, float y, float z) {
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glBegin(GL_POLYGON);
        glVertex3f(x, y, z);
        glVertex3f(-x, y, z);
        glVertex3f(-x, -y, z);
        glVertex3f(x, -y, z);
        glEnd();
    }

    private static void drawAxis() {
        glPushMatrix();
        glLineWidth(3.0f);
        glColor3f(1, 0, 0);
        drawLine(0, 0, 0, AXIS_SIZE, 0, 0);
        glColor3f(0, 1, 0);
        drawLine(0, 0, 0, 0, AXIS_SIZE, 0);
        glColor3f(0, 0, 1);
        drawLine(0, 0, 0, 0, 0, AXIS_SIZE);
        glPopMatrix();
    }

    private static void drawGrid() {
        glPushMatrix();
        glLineWidth(1.0f);
        glColor3f(0.2f, 0.2f, 0.2f);
        for (int i = -GRID_SiZE; i <= GRID_SiZE; i++) {
            drawLine(-20.0f, 0.0f, i, 20.0f, 0.0f, i);
            drawLine(i, 0.0f, -20.0f, i, 0.0f, 20.0f);
        }
        glPopMatrix();
    }

    private static void drawLine(float x1, float y1, float z1, float x2, float y2, float z2) {
        glBegin(GL_LINES);
        glVertex3f(x1, y1, z1);
        glVertex3f(x2, y2, z2);
        glEnd();
    }


}
