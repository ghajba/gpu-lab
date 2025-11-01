package hu.hajba.gpu;

import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.WString;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class DllPath {

    private static final String CACHE_DIR_PROPERTY = "DJL_CACHE_DIR";

    interface Kernel32 extends Library {
        Kernel32 INSTANCE = Native.load("Kernel32", Kernel32.class);

        boolean SetDllDirectoryW(WString path);
    }

    public static void prependDjlCudaToDllSearchPath() {
        String cache = System.getProperty(CACHE_DIR_PROPERTY, Paths.get(System.getProperty("user.home"), ".djl.ai").toString());

        Path pytorchRoot = Paths.get(cache, "pytorch");

        Path chosen = null;
        try (var dirs = Files.list(pytorchRoot)) {
            chosen = dirs.filter(Files::isDirectory)
                    .filter(p -> p.getFileName().toString().contains("cu124-win-x86_64")).min((a, b) -> {
                        try {
                            return Files.getLastModifiedTime(b).compareTo(Files.getLastModifiedTime(a));
                        } catch (Exception e) {
                            return 0;
                        }
                    }).orElse(null);
        } catch (Exception ignore) {
        }

        if (chosen != null) {
            boolean ok = Kernel32.INSTANCE.SetDllDirectoryW(new WString(chosen.toString()));
            if (!ok) {
                System.err.println("WARNING: SetDllDirectory failed for " + chosen);
            }
        }
    }
}
