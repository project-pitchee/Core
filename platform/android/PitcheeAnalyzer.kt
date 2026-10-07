package space.pitchee.core

import java.io.Closeable

enum class ScoreProfile(val value: Int) {
    FEMINIZATION(0),
    MASCULINIZATION(1),
}

class PitcheeAnalyzer private constructor(private var nativeHandle: Long) : Closeable {
    fun analyze(
        samples: FloatArray,
        sampleRate: Int,
        channels: Int,
        scoreProfile: ScoreProfile,
    ): String = nativeAnalyze(
        nativeHandle,
        samples,
        sampleRate,
        channels,
        scoreProfile.value,
    )

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
        channels: Int,
        scoreProfile: Int,
    ): String

    companion object {
        init {
            System.loadLibrary("pitchee_core_jni")
        }

        fun create(
            modelDirectory: String,
            threads: Int = 2,
            minF0Hz: Float = 75f,
            maxF0Hz: Float = 600f,
            minConfidence: Float = 0.9f,
        ): PitcheeAnalyzer {
            return PitcheeAnalyzer(
                nativeCreate(
                    modelDirectory,
                    threads,
                    minF0Hz,
                    maxF0Hz,
                    minConfidence,
                )
            )
        }

        @JvmStatic
        private external fun nativeCreate(
            modelDirectory: String,
            threads: Int,
            minF0Hz: Float,
            maxF0Hz: Float,
            minConfidence: Float,
        ): Long
    }
}
