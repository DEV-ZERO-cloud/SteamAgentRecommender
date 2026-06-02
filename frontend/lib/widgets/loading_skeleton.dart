import 'package:flutter/material.dart';
import '../theme/app_theme.dart';

class LoadingSkeleton extends StatelessWidget {
  const LoadingSkeleton({super.key});

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: 340,
      child: Card(
        margin: const EdgeInsets.all(8),
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _shimmer(150, 20),
              const SizedBox(height: 16),
              _shimmer(double.infinity, 10),
              const SizedBox(height: 8),
              _shimmer(double.infinity, 10),
              const SizedBox(height: 16),
              _shimmer(120, 14),
              const SizedBox(height: 8),
              _shimmer(100, 14),
              const SizedBox(height: 16),
              _shimmer(double.infinity, 36),
            ],
          ),
        ),
      ),
    );
  }

  Widget _shimmer(double width, double height) {
    return Container(
      width: width,
      height: height,
      decoration: BoxDecoration(
        color: AppTheme.steelGrey.withValues(alpha: 0.3),
        borderRadius: BorderRadius.circular(8),
      ),
    );
  }
}
