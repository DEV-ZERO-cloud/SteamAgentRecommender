import 'package:flutter/material.dart';
import '../theme/app_theme.dart';

class AffinityBar extends StatelessWidget {
  final double score;

  const AffinityBar({super.key, required this.score});

  @override
  Widget build(BuildContext context) {
    final clamped = score.clamp(0.0, 100.0);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text('Afinidad',
                style: Theme.of(context).textTheme.bodyMedium),
            Text('${clamped.round()}%',
                style: Theme.of(context).textTheme.labelLarge),
          ],
        ),
        const SizedBox(height: 6),
        ClipRRect(
          borderRadius: BorderRadius.circular(6),
          child: Container(
            height: 10,
            decoration: BoxDecoration(
              color: AppTheme.steelGrey,
              borderRadius: BorderRadius.circular(6),
            ),
            child: FractionallySizedBox(
              widthFactor: clamped / 100,
              child: Container(
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(6),
                  gradient: const LinearGradient(
                    colors: [
                      AppTheme.affinityStart,
                      AppTheme.affinityEnd,
                    ],
                  ),
                ),
              ),
            ),
          ),
        ),
      ],
    );
  }
}
