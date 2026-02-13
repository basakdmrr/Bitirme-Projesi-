import 'package:flutter/material.dart';
import 'pages/role_select.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Hastalık Analiz App',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: const RoleSelectScreen(), // 🔥 Artık buradan başlıyor
    );
  }
}

