import 'dart:convert';
import 'package:http/http.dart' as http;

class HistoryService {
  static const String _baseUrl = "http://10.0.2.2:8000";

  Future<List<Map<String, dynamic>>> getMyRecords(String token) async {
    try {
      final uri = Uri.parse("$_baseUrl/records/");
      
      final response = await http.get(
        uri,
        headers: {
          'Authorization': 'Bearer $token',
          'Accept': 'application/json',
        },
      );

      if (response.statusCode == 200) {
        final List<dynamic> data = jsonDecode(response.body);
        return data.cast<Map<String, dynamic>>();
      } else {
        throw Exception("Kayıtlar alınamadı: ${response.statusCode}");
      }
    } catch (e) {
      throw Exception("Bağlantı hatası: $e");
    }
  }
}
