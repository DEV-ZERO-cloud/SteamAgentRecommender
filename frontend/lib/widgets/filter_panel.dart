import 'package:flutter/material.dart';
import '../providers/recommendation_provider.dart';
import '../theme/app_theme.dart';

class FilterPanel extends StatelessWidget {
  final FilterParams filters;
  final ValueChanged<FilterParams>? onChanged;

  const FilterPanel({
    super.key,
    required this.filters,
    this.onChanged,
  });

  FilterParams _updated({
    double? maxPrice,
    double? minRating,
    int? minYear,
    int? maxYear,
    double? minRecommendations,
  }) {
    return filters.copyWith(
      maxPrice: maxPrice,
      minRating: minRating,
      minYear: minYear,
      maxYear: maxYear,
      minRecommendations: minRecommendations,
    );
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppTheme.surface,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: AppTheme.cardBorder),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('Filtros',
              style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                    fontWeight: FontWeight.w600,
                  )),
          const SizedBox(height: 16),
          _buildSlider(
            context,
            label: 'Precio máximo',
            value: filters.maxPrice,
            min: 0,
            max: 100,
            displayValue: '\$${filters.maxPrice.toStringAsFixed(0)}',
            onChanged: (v) => onChanged?.call(_updated(maxPrice: v)),
          ),
          const SizedBox(height: 12),
          _buildSlider(
            context,
            label: 'Rating mínimo',
            value: filters.minRating,
            min: 0,
            max: 10,
            displayValue: filters.minRating.toStringAsFixed(1),
            onChanged: (v) => onChanged?.call(_updated(minRating: v)),
          ),
          const SizedBox(height: 12),
          _buildSlider(
            context,
            label: 'Año mínimo',
            value: (filters.minYear ?? 1990).toDouble(),
            min: 1990,
            max: 2030,
            displayValue: '${filters.minYear ?? 1990}',
            onChanged: (v) =>
                onChanged?.call(_updated(minYear: v.round())),
          ),
          const SizedBox(height: 12),
          _buildSlider(
            context,
            label: 'Recomendaciones mín.',
            value: filters.minRecommendations,
            min: 0,
            max: 100000,
            displayValue:
                filters.minRecommendations.toStringAsFixed(0),
            divisions: 100,
            onChanged: (v) =>
                onChanged?.call(_updated(minRecommendations: v)),
          ),
        ],
      ),
    );
  }

  Widget _buildSlider(
    BuildContext context, {
    required String label,
    required double value,
    required double min,
    required double max,
    required String displayValue,
    int? divisions,
    required ValueChanged<double> onChanged,
  }) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(label,
                style: const TextStyle(
                    color: AppTheme.textSecondary, fontSize: 13)),
            Text(displayValue,
                style: const TextStyle(
                    color: AppTheme.accent,
                    fontSize: 13,
                    fontWeight: FontWeight.w600)),
          ],
        ),
        SliderTheme(
          data: SliderTheme.of(context).copyWith(
            activeTrackColor: AppTheme.accent,
            inactiveTrackColor: AppTheme.steelGrey,
            thumbColor: AppTheme.accent,
            overlayColor: AppTheme.accent.withValues(alpha: 0.1),
          ),
          child: Slider(
            value: value.clamp(min, max),
            min: min,
            max: max,
            divisions: divisions ?? (max - min).round(),
            onChanged: onChanged,
          ),
        ),
      ],
    );
  }
}
