package space.pitchee.core

import java.io.Closeable

class PitcheeAnalyzer private constructor(private var nativeHandle: Long) : Closeable {
    fun analyze(
        samples: FloatArray,
        sampleRate: Int,
        channels: Int
    ): String = nativeAnalyze(nativeHandle, samples, sampleRate, channels)

    override fun close() {
        if (nativeHandle != 0L) {
            nativeDestroy(nativeHandle)
            nativeHandle = 0L
        }
    }

    private external fun nativeDestroy(handle: Long)
    private external fun nativeAnalyze(
        handle: Long,
        samples: FloatArray,
        sampleRate: Int,
        channels: Int
    ): String

    companion object {
        init {
            System.loadLibrary("pitchee_core_jni")
        }

        fun create(modelDirectory: String, threads: Int = 2): PitcheeAnalyzer {
            return PitcheeAnalyzer(nativeCreate(modelDirectory, threads))
        }

        @JvmStatic
        private external fun nativeCreate(modelDirectory: String, threads: Int): Long
    }
}
