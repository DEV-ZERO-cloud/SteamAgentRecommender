class GameScoreData {
  final int appId;
  final String name;
  final double semanticScore;
  final double parameterScore;
  final int dateFlag;
  final int priceFlag;
  final int positiveRateFlag;
  final int recommendationsFlag;

  GameScoreData({
    required this.appId,
    required this.name,
    required this.semanticScore,
    required this.parameterScore,
    required this.dateFlag,
    required this.priceFlag,
    required this.positiveRateFlag,
    required this.recommendationsFlag,
  });

  factory GameScoreData.fromJson(Map<String, dynamic> json) {
    return GameScoreData(
      appId: json['app_id'] as int,
      name: json['name'] as String,
      semanticScore: (json['semantic_score'] as num).toDouble(),
      parameterScore: (json['parameter_score'] as num).toDouble(),
      dateFlag: json['date_flag'] as int,
      priceFlag: json['price_flag'] as int,
      positiveRateFlag: json['positive_rate_flag'] as int,
      recommendationsFlag: json['recommendations_flag'] as int,
    );
  }
}

class Recommendation {
  final GameScoreData gameScore;
  final List<String> explanations;
  final String? priceTier;
  final String? ratingTier;
  final double tagOverlap;
  final bool isRpg;

  Recommendation({
    required this.gameScore,
    required this.explanations,
    this.priceTier,
    this.ratingTier,
    required this.tagOverlap,
    required this.isRpg,
  });

  factory Recommendation.fromJson(Map<String, dynamic> json) {
    return Recommendation(
      gameScore: GameScoreData.fromJson(
          json['game_score'] as Map<String, dynamic>),
      explanations: (json['explanations'] as List<dynamic>)
          .map((e) => e as String)
          .toList(),
      priceTier: json['price_tier'] as String?,
      ratingTier: json['rating_tier'] as String?,
      tagOverlap: (json['tag_overlap'] as num).toDouble(),
      isRpg: json['is_rpg'] as bool,
    );
  }
}
