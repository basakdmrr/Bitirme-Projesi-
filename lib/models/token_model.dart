import 'user_model.dart';

class Token {
  final String accessToken;
  final String tokenType;
  final UserModel user;

  Token({
    required this.accessToken,
    required this.tokenType,
    required this.user,
  });

  factory Token.fromJson(Map<String, dynamic> json) {
    return Token(
      accessToken: json['access_token'],
      tokenType: json['token_type'],
      user: UserModel.fromJson(json['user']),
    );
  }
}
