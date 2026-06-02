import 'package:flutter/material.dart';
import '../theme/app_theme.dart';

class TagBadge extends StatelessWidget {
  final String label;

  const TagBadge({super.key, required this.label});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(
        color: AppTheme.steelGrey.withValues(alpha: 0.5),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(
          color: AppTheme.cardBorder,
          width: 0.5,
        ),
      ),
      child: Text(
        label,
        style: TextStyle(
          color: AppTheme.accent.withValues(alpha: 0.9),
          fontSize: 12,
          fontWeight: FontWeight.w500,
        ),
      ),
    );
  }
}
