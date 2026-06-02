import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';
import '../models/recommendation.dart';
import '../models/game.dart';
import '../theme/app_theme.dart';
import 'affinity_bar.dart';
import 'tag_badge.dart';

class GameCard extends StatelessWidget {
  final Recommendation recommendation;
  final Game? details;

  const GameCard({
    super.key,
    required this.recommendation,
    this.details,
  });

  @override
  Widget build(BuildContext context) {
    final score = recommendation.gameScore;
    final affinity = recommendation.tagOverlap * 100;

    return Card(
      margin: const EdgeInsets.all(8),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Expanded(
                  child: Text(
                    score.name,
                    style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                          fontWeight: FontWeight.w600,
                        ),
                    maxLines: 2,
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
                if (recommendation.isRpg)
                  Container(
                    padding: const EdgeInsets.symmetric(
                        horizontal: 8, vertical: 2),
                    decoration: BoxDecoration(
                      color: AppTheme.accent.withValues(alpha: 0.2),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: const Text(
                      'RPG',
                      style: TextStyle(
                        color: AppTheme.accent,
                        fontSize: 11,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ),
              ],
            ),
            const SizedBox(height: 12),
            AffinityBar(score: affinity),
            if (details != null) ...[
              const SizedBox(height: 12),
              _buildInfoRow(
                  Icons.star, '${details!.rating.toStringAsFixed(1)} / 10'),
              _buildInfoRow(
                  Icons.attach_money,
                  '\$${details!.price.toStringAsFixed(2)}'),
              if (details!.releaseYear != null)
                _buildInfoRow(
                    Icons.calendar_today, '${details!.releaseYear}'),
            ],
            if (recommendation.explanations.isNotEmpty) ...[
              const SizedBox(height: 10),
              ...recommendation.explanations.map(_buildExplanation),
            ],
            const SizedBox(height: 12),
            SizedBox(
              width: double.infinity,
              child: OutlinedButton.icon(
                onPressed: () {
                  final url = Uri.parse(
                      'https://store.steampowered.com/app/${score.appId}');
                  launchUrl(url, mode: LaunchMode.externalApplication);
                },
                icon: const Icon(Icons.open_in_new, size: 16),
                label: const Text('Abrir en Steam'),
                style: OutlinedButton.styleFrom(
                  foregroundColor: AppTheme.accent,
                  side: const BorderSide(color: AppTheme.accent),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(8),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildInfoRow(IconData icon, String text) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2),
      child: Row(
        children: [
          Icon(icon, size: 14, color: AppTheme.textSecondary),
          const SizedBox(width: 6),
          Text(text,
              style: const TextStyle(
                  color: AppTheme.textSecondary, fontSize: 13)),
        ],
      ),
    );
  }

  Widget _buildExplanation(String explanation) {
    final colonIdx = explanation.indexOf(':');
    if (colonIdx >= 0 && colonIdx < explanation.length - 1) {
      final before = explanation.substring(0, colonIdx).trim();
      final after = explanation.substring(colonIdx + 1).trim();
      final allTags = after.split(', ').where((t) => t.isNotEmpty).toList();

      if (allTags.isNotEmpty) {
        const maxVisible = 4;
        final visible = allTags.take(maxVisible).toList();
        final remaining = allTags.length - maxVisible;
        final children = visible.map((t) => TagBadge(label: t)).toList();
        if (remaining > 0) {
          children.add(TagBadge(label: '+$remaining más'));
        }

        return Padding(
          padding: const EdgeInsets.only(bottom: 8),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                before,
                style: const TextStyle(
                  color: AppTheme.textSecondary,
                  fontSize: 12,
                  fontWeight: FontWeight.w500,
                ),
              ),
              const SizedBox(height: 6),
              Wrap(
                spacing: 6,
                runSpacing: 4,
                children: children,
              ),
            ],
          ),
        );
      }
    }

    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: TagBadge(label: explanation),
    );
  }
}
