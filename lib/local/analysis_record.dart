class AnalysisRecord {
  final int? id;
  final String createdAtIso;
  final String label;
  final double confidence;
  final String audioPath;

  AnalysisRecord({
    this.id,
    required this.createdAtIso,
    required this.label,
    required this.confidence,
    required this.audioPath,
  });

  Map<String, Object?> toMap() => {
        "id": id,
        "createdAtIso": createdAtIso,
        "label": label,
        "confidence": confidence,
        "audioPath": audioPath,
      };

  factory AnalysisRecord.fromMap(Map<String, Object?> m) {
    return AnalysisRecord(
      id: m["id"] as int?,
      createdAtIso: m["createdAtIso"] as String,
      label: m["label"] as String,
      confidence: (m["confidence"] as num).toDouble(),
      audioPath: m["audioPath"] as String,
    );
  }
}
