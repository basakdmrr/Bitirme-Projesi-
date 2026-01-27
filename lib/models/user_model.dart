class UserModel {
  final int id;
  final String tc;
  final String name;
  final DateTime createdAt;

  const UserModel({
    required this.id,
    required this.tc,
    required this.name,
    required this.createdAt,
  });

  factory UserModel.fromJson(Map<String, dynamic> json) {
    return UserModel(
      id: json['id'],
      tc: json['tc'],
      name: json['name'],
      createdAt: DateTime.tryParse(json['created_at'] ?? '') ??
          DateTime.fromMillisecondsSinceEpoch(0),
    );
  }

  Map<String, dynamic> toJson() => {
        "id": id,
        "tc": tc,
        "name": name,
        "created_at": createdAt.toIso8601String(),
      };

  UserModel copyWith({
    int? id,
    String? tc,
    String? name,
    DateTime? createdAt,
  }) {
    return UserModel(
      id: id ?? this.id,
      tc: tc ?? this.tc,
      name: name ?? this.name,
      createdAt: createdAt ?? this.createdAt,
    );
  }
}
