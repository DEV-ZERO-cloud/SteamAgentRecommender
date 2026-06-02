import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../providers/recommendation_provider.dart';
import '../theme/app_theme.dart';
import '../widgets/steam_profile_input.dart';
import '../widgets/game_card.dart';
import '../widgets/filter_panel.dart';
import '../widgets/loading_skeleton.dart';

class HomeScreen extends ConsumerWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final appState = ref.watch(recommendationProvider);
    final isWide = MediaQuery.of(context).size.width > 900;

    return Scaffold(
      body: SafeArea(
        child: Column(
          children: [
            _buildHeader(context),
            const SteamProfileInput(),
            const SizedBox(height: 8),
            _buildStatus(context, appState),
            if (appState.state == AppState.loaded ||
                appState.state == AppState.loadingGames)
              Expanded(
                child: _buildContent(context, appState, isWide, ref),
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildHeader(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 20),
      decoration: const BoxDecoration(
        gradient: LinearGradient(
          colors: [Color(0xFF000000), AppTheme.background],
          begin: Alignment.topCenter,
          end: Alignment.bottomCenter,
        ),
      ),
      child: Row(
        children: [
          const Icon(Icons.videogame_asset, color: AppTheme.accent, size: 32),
          const SizedBox(width: 12),
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'SteamAgent',
                style: Theme.of(context).textTheme.headlineLarge?.copyWith(
                      color: AppTheme.accent,
                      letterSpacing: 1.5,
                    ),
              ),
              Text(
                'Recomendador Inteligente de RPGs',
                style: Theme.of(context).textTheme.bodyMedium,
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildStatus(BuildContext context, RecommendationState state) {
    switch (state.state) {
      case AppState.idle:
        return Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            children: [
              Icon(Icons.search,
                  size: 64,
                  color: AppTheme.textSecondary.withValues(alpha: 0.3)),
              const SizedBox(height: 16),
              Text(
                'Ingresa tu SteamID para obtener recomendaciones personalizadas',
                textAlign: TextAlign.center,
                style: Theme.of(context).textTheme.bodyMedium,
              ),
            ],
          ),
        );
      case AppState.loadingProfile:
        return const Padding(
          padding: EdgeInsets.all(16),
          child: LinearProgressIndicator(
            color: AppTheme.accent,
            backgroundColor: AppTheme.steelGrey,
          ),
        );
      case AppState.loadingGames:
        return Expanded(
          child: LayoutBuilder(
            builder: (context, constraints) {
              final cols = constraints.maxWidth > 900
                  ? 3
                  : constraints.maxWidth > 600
                      ? 2
                      : 1;
              return GridView.builder(
                padding: const EdgeInsets.all(12),
                gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
                  crossAxisCount: cols,
                  mainAxisSpacing: 8,
                  crossAxisSpacing: 8,
                  childAspectRatio: 0.75,
                ),
                itemCount: cols * 2,
                itemBuilder: (_, __) => const LoadingSkeleton(),
              );
            },
          ),
        );
      case AppState.error:
        return Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            children: [
              const Icon(Icons.error_outline,
                  size: 48, color: Colors.redAccent),
              const SizedBox(height: 12),
              Text(
                state.error ?? 'Error desconocido',
                textAlign: TextAlign.center,
                style: const TextStyle(color: Colors.redAccent),
              ),
            ],
          ),
        );
      default:
        return const SizedBox.shrink();
    }
  }

  Widget _buildContent(
    BuildContext context,
    RecommendationState state,
    bool isWide,
    WidgetRef ref,
  ) {
    if (isWide) {
      return Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 280,
            child: SingleChildScrollView(
              padding: const EdgeInsets.all(16),
              child: FilterPanel(
                filters: state.filters,
                onChanged: (f) =>
                    ref.read(recommendationProvider.notifier).updateFilters(f),
              ),
            ),
          ),
          Expanded(child: _buildGrid(context, state)),
        ],
      );
    }

    return Column(
      children: [
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16),
          child: SizedBox(
            width: double.infinity,
            child: OutlinedButton.icon(
              onPressed: () => _showFilterSheet(context, state, ref),
              icon: const Icon(Icons.filter_list),
              label: const Text('Filtros'),
              style: OutlinedButton.styleFrom(
                foregroundColor: AppTheme.accent,
                side: const BorderSide(color: AppTheme.cardBorder),
              ),
            ),
          ),
        ),
        Expanded(child: _buildGrid(context, state)),
      ],
    );
  }

  void _showFilterSheet(
      BuildContext context, RecommendationState state, WidgetRef ref) {
    showModalBottomSheet(
      context: context,
      backgroundColor: AppTheme.surface,
      shape: const RoundedRectangleBorder(
        borderRadius:
            BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (_) => Padding(
        padding: const EdgeInsets.all(24),
        child: FilterPanel(
          filters: state.filters,
          onChanged: (f) =>
              ref.read(recommendationProvider.notifier).updateFilters(f),
        ),
      ),
    );
  }

  Widget _buildGrid(BuildContext context, RecommendationState state) {
    final recs = state.filteredRecommendations;
    final details = state.gameDetails;

    if (recs.isEmpty) {
      return const Center(
        child: Text('No se encontraron juegos con esos filtros.',
            style: TextStyle(color: AppTheme.textSecondary)),
      );
    }

    return LayoutBuilder(
      builder: (context, constraints) {
        final crossAxisCount = constraints.maxWidth > 900
            ? 3
            : constraints.maxWidth > 600
                ? 2
                : 1;
        return GridView.builder(
          padding: const EdgeInsets.all(12),
          gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
            crossAxisCount: crossAxisCount,
            mainAxisSpacing: 8,
            crossAxisSpacing: 8,
            childAspectRatio: 0.65,
          ),
          itemCount: recs.length,
          itemBuilder: (context, index) {
            final rec = recs[index];
            return GameCard(
              recommendation: rec,
              details: details[rec.gameScore.appId],
            );
          },
        );
      },
    );
  }
}
