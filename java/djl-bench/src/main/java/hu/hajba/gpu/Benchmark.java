package hu.hajba.gpu;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import picocli.CommandLine;
import picocli.CommandLine.Option;

import java.util.concurrent.Callable;

@CommandLine.Command(
    name = "gpu-bench",
    mixinStandardHelpOptions = true,
    version = "gpu-bench 0.1",
    description = "Deterministic NDArray matmul benchmark for CPU/GPU with DJL."
)
public class Benchmark implements Callable<Integer> {

    @Option(names = {"-n", "--size"}, paramLabel = "N", description = "Matrix size NxN.", defaultValue = "4096")
    int n;

    @Option(names = {"-w", "--warmup"}, paramLabel = "W", description = "Warmup iterations (not timed).", defaultValue = "10")
    int warmup;

    @Option(names = {"-i", "--iterations"}, paramLabel = "I", description = "Timed iterations.", defaultValue = "50")
    int iterations;

    @Option(names = {"-d", "--device"}, paramLabel = "DEV", description = "Device: auto|cpu|gpu", defaultValue = "auto")
    String deviceOpt;

    @Option(names = {"-e", "--engine"}, paramLabel = "ENG", description = "Force DJL engine (pytorch|tensorflow|onnxruntime). Uses classpath by default.")
    String engineOpt;

    @Option(names = {"--seed"}, paramLabel = "SEED", description = "Deterministic data seed (ignored for generated pattern).", defaultValue = "42")
    long seed;

    @Option(names = {"--pattern"}, description = "Use deterministic pattern tensors (default) instead of RNG.", defaultValue = "true")
    boolean pattern;

    @Option(names = {"--csv"}, paramLabel = "FILE", description = "Append results as CSV to FILE (op,engine,device,n,warm,iters,ms).")
    String csvFile;

    static void main(String[] args) {
        DllPath.prependDjlCudaToDllSearchPath();

        int rc = new CommandLine(new Benchmark()).execute(args);
        System.exit(rc);
    }

    @Override
    public Integer call() {
        if (engineOpt != null && !engineOpt.isBlank()) {
            System.setProperty("ai.djl.default_engine", engineOpt);
        }
        String engine = Engine.getInstance().getEngineName();
        Device device = chooseDevice(deviceOpt);

        System.out.printf("Engine=%s  Device=%s  N=%d  warmup=%d  iters=%d%n", engine, device, n, warmup, iterations);

        double ms = runBenchmark(device, n, warmup, iterations, pattern, seed);

        System.out.printf("Avg: %.2f ms%n", ms);
        if (csvFile != null && !csvFile.isBlank()) {
            appendCsv(csvFile, "dot", engine, device, n, warmup, iterations, ms);
        }
        // if auto, also compare the other device for convenience
        if ("auto".equalsIgnoreCase(deviceOpt)) {
            Device other = device.isGpu() ? Device.cpu() : tryGpuOrCpu();
            if (!other.equals(device)) {
                double ms2 = runBenchmark(other, n, warmup, iterations, pattern, seed);
                System.out.printf("Other(%s) Avg: %.2f ms  | Speedup: %.2fx%n",
                    other, ms2, Math.max(ms, ms2) / Math.min(ms, ms2));
            }
        }
        return 0;
    }

    private Device chooseDevice(String opt) {
        return switch (opt.toLowerCase()) {
            case "cpu" -> Device.cpu();
            default -> tryGpuOrCpu();
        };
    }

    private Device tryGpuOrCpu() {
        int gpuCount = Engine.getInstance().getGpuCount();
        if (gpuCount > 0) {
            return Device.gpu();
        }
        return Device.cpu();
    }

    private double runBenchmark(Device device, int n, int warmup, int iterations, boolean pattern, long seed) {
        try (NDManager manager = NDManager.newBaseManager(device)) {
            NDArray a = pattern ? patternA(manager, n) : seeded(manager, n, seed);
            NDArray b = pattern ? patternB(manager, n) : seeded(manager, n, seed ^ 0x9E3779B97F4A7C15L);

            System.out.println("a shape " + a.getShape());

            // warmup
            for (int i = 0; i < warmup; i++) {
                a.dot(b);
            }

            long total = 0;
            for (int i = 0; i < iterations; i++) {
                long t0 = System.nanoTime();
                NDArray c = a.dot(b);
                //                manager.cap(); // barrier for timing
                long t1 = System.nanoTime();
                total += (t1 - t0);

                if (i == 0) {
                    float checksum = c.sum().getFloat();
                    System.out.printf("Checksum(%s): %.4f%n", device, checksum);
                }
            }
            return total / 1e6 / iterations;
        }
    }

    private NDArray patternA(NDManager m, int n) {
        return m.arange(0, (long) n * n, 1, DataType.FLOAT32)
            .reshape(new Shape(n, n))
            .div(n);
    }

    private NDArray patternB(NDManager m, int n) {
        return patternA(m, n).transpose().add(1f);
    }

    private NDArray seeded(NDManager m, int n, long seed) {
        // deterministic “random” via LCG on host, then upload
        float[] data = new float[n * n];
        long x = seed;
        for (int i = 0; i < data.length; i++) {
            x = x * 6364136223846793005L + 1;
            data[i] = ((x >>> 8) & 0xFFFFFF) / (float) (1 << 24);
        }
        return m.create(data).reshape(n, n);
    }

    private void appendCsv(String path, String operation, String engine, Device device,
        int n, int warm, int iterations, double ms) {
        String line = String.format("%s,%s,%s,%d,%d,%d,%.4f%n",
            operation, engine, device, n, warm, iterations, ms);
        try (java.io.FileWriter fw = new java.io.FileWriter(path, true)) {
            if (new java.io.File(path).length() == 0) {
                fw.write("operation,engine,device,n,warmup,iterations,ms\n");
            }
            fw.write(line);
        } catch (Exception e) {
            System.err.println("CSV append failed: " + e.getMessage());
        }
    }
}
