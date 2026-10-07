import CPitcheeCore
import Foundation

public final class PitcheeAnalyzer {
    private var handle: OpaquePointer?
    private let lock = NSLock()

    public init(
        modelDirectory: URL,
        threads: Int32 = 2,
        minF0Hz: Float = 75,
        maxF0Hz: Float = 600,
        minConfidence: Float = 0.9
    ) throws {
        var options = pitchee_analyzer_options_t(
            intra_op_threads: threads,
            use_coreml: 1,
            thresholds: pitchee_f0_thresholds_t(
                min_f0_hz: minF0Hz,
                max_f0_hz: maxF0Hz,
                min_confidence: minConfidence,
                reserved: 0
            )
        )
        var error = [CChar](repeating: 0, count: 1024)
        var created: OpaquePointer?
        let status = pitchee_analyzer_create(
            modelDirectory.path,
            &options,
            &created,
            &error,
            error.count
        )
        guard status == PITCHEE_SUCCESS, let created else {
            throw NSError(
                domain: "PitcheeCore",
                code: Int(status.rawValue),
                userInfo: [NSLocalizedDescriptionKey: String(cString: error)]
            )
        }
        handle = created
    }

    deinit {
        if let handle {
            pitchee_analyzer_destroy(handle)
        }
    }

    public func analyze(
        samples: [Float],
        sampleRate: Int32,
        channels: Int32,
        scoreProfile: pitchee_score_profile_t
    ) throws -> String {
        lock.lock()
        defer { lock.unlock() }
        guard let handle else {
            throw NSError(domain: "PitcheeCore", code: -1)
        }

        var output: UnsafeMutablePointer<CChar>?
        var error = [CChar](repeating: 0, count: 1024)
        let status = samples.withUnsafeBufferPointer { buffer in
            pitchee_analyzer_analyze_pcm(
                handle,
                buffer.baseAddress,
                buffer.count,
                sampleRate,
                channels,
                scoreProfile,
                nil,
                nil,
                &output,
                &error,
                error.count
            )
        }
        guard status == PITCHEE_SUCCESS, let output else {
            throw NSError(
                domain: "PitcheeCore",
                code: Int(status.rawValue),
                userInfo: [NSLocalizedDescriptionKey: String(cString: error)]
            )
        }
        defer { pitchee_string_free(output) }
        return String(cString: output)
    }
}
