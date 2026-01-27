class VoiceRecord {
  final int id;
  final int userId;
  final String? tc;
  final String filePath;
  final DateTime? createdAt;
  final String? predictionResult;
  final double? confidenceScore;
  final String processingStatus;

  const VoiceRecord({
    required this.id,
    required this.userId,
    this.tc,
    required this.filePath,
    this.createdAt,
    this.predictionResult,
    this.confidenceScore,
    this.processingStatus = 'pending',
  });

  factory VoiceRecord.fromJson(Map<String, dynamic> json) {
    DateTime? parsedDate;
    try {
      if (json['created_at'] != null) {
        parsedDate = DateTime.parse(json['created_at'].toString());
      }
    } catch (_) {
      parsedDate = null;
    }
    return VoiceRecord(
      id: json['id'] is int ? json['id'] : int.tryParse(json['id'].toString()) ?? 0,
      userId: json['user_id'] is int ? json['user_id'] : int.tryParse(json['user_id'].toString()) ?? 0,
      tc: json['tc']?.toString(),
      filePath: json['file_path']?.toString() ?? '',
      createdAt: parsedDate,
      predictionResult: json['prediction_result']?.toString(),
      confidenceScore: json['confidence_score'] != null ? double.tryParse(json['confidence_score'].toString()) : null,
      processingStatus: json['processing_status']?.toString() ?? 'pending',
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'user_id': userId,
      'tc': tc,
      'file_path': filePath,
      'created_at': createdAt?.toIso8601String(),
      'prediction_result': predictionResult,
      'confidence_score': confidenceScore,
      'processing_status': processingStatus,
    };
  }

  VoiceRecord copyWith({
    int? id,
    int? userId,
    String? tc,
    String? filePath,
    DateTime? createdAt,
    String? predictionResult,
    double? confidenceScore,
    String? processingStatus
  }) {
    return VoiceRecord(
      id: id ?? this.id,
      userId: userId ?? this.userId, 
      tc: tc ?? this.tc, 
      filePath: filePath ?? this.filePath, 
      createdAt: createdAt ?? this.createdAt, 
      predictionResult: predictionResult ?? this.predictionResult, 
      confidenceScore: confidenceScore ?? this.confidenceScore, 
      processingStatus: processingStatus ?? this.processingStatus
   );
 }
}