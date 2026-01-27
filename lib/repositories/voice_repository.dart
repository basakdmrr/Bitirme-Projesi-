import 'dart:io';

import '../models/voice_record_model.dart';
import '../services/predict_service.dart';
import '../services/auth_service.dart';

class VoiceRepository {
  final PredictService _predictService;

  VoiceRepository({PredictService? predictService})
      : _predictService = predictService ?? PredictService();

  /// 🎤 Ses yükle + tahmin al
  Future<VoiceRecord> uploadAndPredict(File audioFile) async {
    if (!AuthService.isLoggedIn) {
      throw Exception("Kullanıcı giriş yapmamış");
    }

    final token = AuthService.token!.accessToken;

    return await _predictService.sendAudioForPrediction(
      audioFile: audioFile,
      token: token,
    );
  }
}
