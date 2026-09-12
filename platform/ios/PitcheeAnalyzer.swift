import CPitcheeCore
import Foundation

public final class PitcheeAnalyzer {
    private var handle: OpaquePointer?
    private let lock = NSLock()

    public init(modelDirectory: URL, threads: Int32 = 2) throws {
        var options = pitchee_analyzer_options_t(
            intra_op_threads: threads,
            use_coreml: 1,
            reserved: 0
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
        channels: Int32
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
