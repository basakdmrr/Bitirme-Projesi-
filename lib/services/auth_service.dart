import 'dart:convert';
import 'package:http/http.dart' as http;
import '../models/token_model.dart';

class AuthService {
  /// 🔑 GLOBAL TOKEN (şimdilik memory'de)
  static Token? token;

  final String baseUrl = "http://10.0.2.2:8000";

  /// 🔑 Authorization header (korumalı endpoint'ler için)
  static Map<String, String> get authHeader {
    if (token == null) return {};
    return {
      "Authorization": "${token!.tokenType} ${token!.accessToken}",
      "Content-Type": "application/json",
    };
  }

  /// ✅ LOGIN
  Future<Token> login(String tc, String password) async {
    final response = await http.post(
      Uri.parse("$baseUrl/auth/login"),
      headers: const {
        "Content-Type": "application/json",
      },
      body: jsonEncode({
        "tc": tc,
        "password": password,
      }),
    );

    if (response.statusCode == 200) {
      final t = Token.fromJson(jsonDecode(response.body));
      token = t; // 🔴 TOKEN SET
      return t;
    } else {
      throw Exception("Giriş başarısız: ${response.body}");
    }
  }

  /// ✅ REGISTER
  Future<Token> register(String tc, String name, String password) async {
    final response = await http.post(
      Uri.parse("$baseUrl/auth/register"),
      headers: const {
        "Content-Type": "application/json",
      },
      body: jsonEncode({
        "tc": tc,
        "name": name,
        "password": password,
      }),
    );

    if (response.statusCode == 200 || response.statusCode == 201) {
      final t = Token.fromJson(jsonDecode(response.body));
      token = t; // 🔴 TOKEN SET
      return t;
    } else {
      throw Exception("Kayıt başarısız: ${response.body}");
    }
  }

  /// 🚪 LOGOUT
  static void logout() {
    token = null;
  }

  /// 🔍 Giriş yapılmış mı?
  static bool get isLoggedIn => token != null;
}
