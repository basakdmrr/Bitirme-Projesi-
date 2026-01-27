import '../services/auth_service.dart';
import '../services/predict_service.dart';
import '../repositories/voice_repository.dart';

class AppContainer {
  static final authService = AuthService();
  static final predictService = PredictService();
  static final voiceRepository =
      VoiceRepository(predictService: predictService);
}
