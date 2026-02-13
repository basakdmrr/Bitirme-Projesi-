import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';
import '../models/token_model.dart';

class AuthService {
  static Token? token;
  final String baseUrl = "http://10.0.2.2:8000";

  static const _tokenKey = "auth_token";

  static Map<String, String> get authHeader {
    if (token == null) return {};
    return {
      "Authorization": "${token!.tokenType} ${token!.accessToken}",
      "Content-Type": "application/json",
    };
  }

  Future<void> saveToken(Token t) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_tokenKey, jsonEncode(t.toJson()));
  }

  Future<void> loadToken() async {
    final prefs = await SharedPreferences.getInstance();
    final data = prefs.getString(_tokenKey);
    if (data != null) {
      token = Token.fromJson(jsonDecode(data));
    }
  }

  Future<void> clearToken() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_tokenKey);
    token = null;
  }

  Future<Token> login(String tc, String password) async {
    final response = await http.post(
      Uri.parse("$baseUrl/auth/login"),
      headers: {"Content-Type": "application/json"},
      body: jsonEncode({"tc": tc, "password": password}),
    );

    if (response.statusCode == 200) {
      final t = Token.fromJson(jsonDecode(response.body));
      token = t;
      await saveToken(t);
      return t;
    } else {
      throw Exception("Giriş başarısız");
    }
  }
 Future<Token> register(String tc, String name, String password) async {
    final response = await http.post(
      Uri.parse("$baseUrl/auth/register"),
      headers: {
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
      token = t;
      return t;
    } else {
      throw Exception("Kayıt başarısız: ${response.body}");
    }
  }

  /// 🚪 LOGOUT
  static void logout() {
    token = null;
  }
  static bool get isLoggedIn => token != null;
}
