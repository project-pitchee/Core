package space.pitchee.benchmark;

import android.app.Activity;
import android.os.Bundle;
import android.view.Gravity;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

public final class MainActivity extends Activity {
    static {
        System.loadLibrary("pitchee_benchmark");
    }

    private TextView output;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        LinearLayout layout = new LinearLayout(this);
        layout.setOrientation(LinearLayout.VERTICAL);
        int padding = (int) (20 * getResources().getDisplayMetrics().density);
        layout.setPadding(padding, padding, padding, padding);

        Button run = new Button(this);
        run.setText("Run ncnn benchmark");
        layout.addView(run, new LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        ));

        output = new TextView(this);
        output.setTextSize(13);
        output.setGravity(Gravity.START);
        output.setText("Preparing models...");
        ScrollView scroll = new ScrollView(this);
        scroll.addView(output);
        layout.addView(scroll, new LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            0,
            1
        ));

        setContentView(layout);
        run.setOnClickListener(view -> runBenchmark());
        runBenchmark();
    }

    private void runBenchmark() {
        output.setText("Preparing models...");
        new Thread(() -> {
            try {
                File models = new File(getFilesDir(), "models");
                File audio = new File(getFilesDir(), "test_audio");
                copyAssetDirectory("models", models);
                copyAssetDirectory("test_audio", audio);
                String report = nativeRunNcnnBenchmark(
                    models.getAbsolutePath(),
                    audio.getAbsolutePath(),
                    4
                );
                runOnUiThread(() -> output.setText(report));
            } catch (Throwable error) {
                runOnUiThread(() -> output.setText(
                    "Benchmark failed:\n" + error
                ));
            }
        }, "pitchee-benchmark").start();
    }

    private void copyAssetDirectory(String assetPath, File target)
        throws IOException {
        String[] children = getAssets().list(assetPath);
        if (children == null || children.length == 0) {
            copyAsset(assetPath, target);
            return;
        }
        if (!target.exists() && !target.mkdirs()) {
            throw new IOException("Unable to create " + target);
        }
        for (String child : children) {
            copyAssetDirectory(assetPath + "/" + child, new File(target, child));
        }
    }

    private void copyAsset(String assetPath, File target) throws IOException {
        File parent = target.getParentFile();
        if (parent != null && !parent.exists() && !parent.mkdirs()) {
            throw new IOException("Unable to create " + parent);
        }
        try (InputStream input = getAssets().open(assetPath);
             FileOutputStream output = new FileOutputStream(target)) {
            byte[] buffer = new byte[1024 * 1024];
            int count;
            while ((count = input.read(buffer)) != -1) {
                output.write(buffer, 0, count);
            }
        }
    }

    private native String nativeRunNcnnBenchmark(
        String modelDirectory,
        String audioDirectory,
        int threads
    );
}
