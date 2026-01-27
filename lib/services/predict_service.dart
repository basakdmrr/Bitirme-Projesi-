import 'dart:io';
import 'package:dio/dio.dart';
import 'package:path/path.dart' as p;

import '../core/constants.dart';
import '../models/voice_record_model.dart';
import '../services/auth_service.dart';

class PredictService {
  final Dio _dio = Dio(
    BaseOptions(
      baseUrl: kBaseUrl,
      connectTimeout: const Duration(seconds: 20),
      receiveTimeout: const Duration(seconds: 90),
    ),
  );

  Future<VoiceRecord> predictWav(File wavFile) async {
    final fileName = p.basename(wavFile.path);

    final formData = FormData.fromMap({
      "file": await MultipartFile.fromFile(
        wavFile.path,
        filename: fileName,
      ),
    });

    final res = await _dio.post(
      kPredictPath,
      data: formData,
      options: Options(
        headers: AuthService.authHeader, // 🔑 TOKEN OTOMATİK
      ),
    );

    final data = res.data;
    if (data is! Map) {
      throw Exception("API JSON bekleniyor. Gelen: $data");
    }

    return VoiceRecord.fromJson(
      Map<String, dynamic>.from(data),
    );
  }
}
