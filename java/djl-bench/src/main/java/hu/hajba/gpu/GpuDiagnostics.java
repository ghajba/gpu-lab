package hu.hajba.gpu;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

public class GpuDiagnostics {

    private static final String CACHE_DIR_PROPERTY = "DJL_CACHE_DIR";

    static void main(String... args) {

        DllPath.prependDjlCudaToDllSearchPath(CACHE_DIR_PROPERTY);

        Engine.debugEnvironment();
        System.out.println("== DJL GPU Diagnostics ==");
        System.out.println("DJL Engine: " + Engine.getInstance().getEngineName());
        System.out.println("DJL Engine version: " + Engine.getInstance().getVersion());
        System.out.println("DJL Version: " + Engine.getDjlVersion());

        int gpuCount = Engine.getInstance().getGpuCount();
        System.out.println("GPU Count: " + gpuCount);

        Device device = gpuCount > 0 ? Device.gpu() : Device.cpu();
        System.out.println("Using device: " + device);

        try (NDManager manager = NDManager.newBaseManager(device)) {
            NDArray arrayA = manager.randomUniform(0f, 1f, new Shape(1024, 1024), DataType.FLOAT32);
            NDArray arrayB = manager.randomUniform(0f, 1f, new Shape(1024, 1024), DataType.FLOAT32);
            NDArray result = arrayA.dot(arrayB);

            System.out.println("Sum: " + result.sum().toFloatArray()[0]);
        }


        System.out.println(Engine.getAllEngines());
        for (Device d : Engine.getInstance().getDevices())
            System.out.println("Device: " + d.toString());
    }
}
