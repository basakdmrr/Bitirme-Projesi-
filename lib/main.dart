import 'package:flutter/material.dart';

import 'app/app_container.dart';
import 'screens/splash_screen.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // 🔗 Service & Repository init
  AppContainer.init();

  runApp(const BreathAIApp());
}

class BreathAIApp extends StatelessWidget {
  const BreathAIApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Breath AI',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        primaryColor: const Color(0xFF0EA5E9),
        scaffoldBackgroundColor: Colors.white,
        fontFamily: 'Roboto',
      ),
      home: const SplashScreen(),
    );
  }
}
