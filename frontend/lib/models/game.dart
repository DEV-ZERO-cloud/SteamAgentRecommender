class Game {
  final int appId;
  final String name;
  final double price;
  final double positiveReviews;
  final double negativeReviews;
  final int totalReviews;
  final double recommendationsRatio;
  final double rating;
  final int? releaseYear;
  final List<String>? platforms;
  final String? shortDescription;
  final List<String> categories;
  final List<String> genres;
  final List<String> tags;

  Game({
    required this.appId,
    required this.name,
    this.price = 0.0,
    this.positiveReviews = 0.0,
    this.negativeReviews = 0.0,
    this.totalReviews = 0,
    this.recommendationsRatio = 0.0,
    this.rating = 0.0,
    this.releaseYear,
    this.platforms,
    this.shortDescription,
    this.categories = const [],
    this.genres = const [],
    this.tags = const [],
  });

  factory Game.fromJson(Map<String, dynamic> json) {
    return Game(
      appId: json['app_id'] as int,
      name: json['name'] as String,
      price: (json['price'] as num?)?.toDouble() ?? 0.0,
      positiveReviews:
          (json['positive_reviews'] as num?)?.toDouble() ?? 0.0,
      negativeReviews:
          (json['negative_reviews'] as num?)?.toDouble() ?? 0.0,
      totalReviews: json['total_reviews'] as int? ?? 0,
      recommendationsRatio:
          (json['recommendations_ratio'] as num?)?.toDouble() ?? 0.0,
      rating: (json['rating'] as num?)?.toDouble() ?? 0.0,
      releaseYear: json['release_year'] as int?,
      platforms: (json['platforms'] as List<dynamic>?)
          ?.map((e) => e as String)
          .toList(),
      shortDescription: json['short_description'] as String?,
      categories: (json['categories'] as List<dynamic>?)
              ?.map((e) => e as String)
              .toList() ??
          [],
      genres: (json['genres'] as List<dynamic>?)
              ?.map((e) => e as String)
              .toList() ??
          [],
      tags: (json['tags'] as List<dynamic>?)
              ?.map((e) => e as String)
              .toList() ??
          [],
    );
  }
}
