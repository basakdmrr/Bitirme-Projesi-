import 'package:flutter/material.dart';

class Bg extends StatelessWidget {
  final Widget child;
  final bool light;

  const Bg({
    super.key,
    required this.child,
    this.light = false,
  });

  @override
  Widget build(BuildContext context) {
    final colors = light
        ? const [
            Color(0xFFE9F7FF),
            Color(0xFFBFEFFF),
            Color(0xFFA7F3D0),
          ]
        : const [
            Color(0xFF0B4A7A),
            Color(0xFF0EA5E9),
            Color(0xFF14B8A6),
          ];

    return Container(
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: colors,
        ),
      ),
      child: child,
    );
  }
}
