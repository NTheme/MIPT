#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace Compare {
static const long double kEpsilon = 1e-10;

bool areEqualLD(long double left, long double right) {
  return std::abs(left - right) < kEpsilon;
}

template <typename Type>
bool areEqualPair(const std::pair<Type, Type>& left,
                  const std::pair<Type, Type>& right) {
  return (left.first == right.first && left.second == right.second) ||
         (left.second == right.first && left.first == right.second);
}
};  // namespace Compare

namespace Angle {
static const long double degreePi = 180.0;

long double toRadians(long double angle) { return angle * M_PIl / degreePi; }

long double toDegrees(long double angle) { return angle * degreePi / M_PIl; }

};  // namespace Angle

class Line;

struct Point {
  Point();
  Point(long double xNew, long double yNew);

  Point operator-() const;
  Point& operator+=(const Point& other);
  Point& operator-=(const Point& other);
  Point& operator*=(long double coeff);
  Point& operator/=(long double coeff);
  bool operator==(const Point& other) const;

  long double distance(const Point& other) const;
  long double scalarMultiply(const Point& other) const;
  long double vectorMultiply(const Point& other) const;
  long double absolute() const;
  bool isCollinearTo(const Point& other) const;

  Point toUnit() const;
  Point rotate90() const;
  Point toPositiveX() const;
  void rotate(const Point& center, long double angle);
  void rotateRadians(const Point& center, long double angle);
  void reflect(const Point& center);
  void reflect(const Line& axis);
  void scale(const Point& center, long double coefficient);

  long double x, y;
};

class Line {
 public:
  Line();
  Line(const Point& first, const Point& second);
  Line(long double coeff, long double shift);
  Line(const Point& point, long double coeff);

  bool operator==(const Line& other) const;

  Point getNormal() const;
  long double getDistance() const;

 private:
  Point normal;
  long double distance;
};

class Shape {
 public:
  Shape();
  virtual ~Shape() = default;

  virtual bool operator==(const Shape& other) const = 0;

  virtual long double area() const;
  virtual long double perimeter() const;
  virtual bool isCongruentTo(const Shape& other) const = 0;
  virtual bool isSimilarTo(const Shape& other) const = 0;
  virtual bool containsPoint(const Point& point) const = 0;

  virtual void rotate(const Point& center, long double angle) = 0;
  virtual void rotateRadians(const Point& center, long double angle) = 0;
  virtual void reflect(const Point& center) = 0;
  virtual void reflect(const Line& axis) = 0;
  virtual void move(const Point& vector) = 0;
  virtual void scale(const Point& center, long double coeff) = 0;

 protected:
  enum class UpdateCond;

  mutable long double areaVal;
  mutable long double perimeterVal;

  mutable UpdateCond initialized;

  virtual void updArea() const = 0;
  virtual void updPerimeter() const = 0;
  virtual void updAll(UpdateCond condition) const;
};

class Polygon : public Shape {
 public:
  Polygon();
  Polygon(const Point& first, const Point& second, const Point& third);
  Polygon(const std::vector<Point>& points);

  template <typename... Args>
  Polygon(Args... points);

  bool operator==(const Polygon& other) const;
  bool operator==(const Shape& other) const final;

  size_t verticesCount() const;
  const std::vector<Point>& getVertices() const;
  bool isConvex() const;

  bool isCongruentTo(const Shape& other) const final;
  bool isSimilarTo(const Shape& other) const final;
  bool containsPoint(const Point& point) const final;

  void rotate(const Point& center, long double angle) final;
  void rotateRadians(const Point& center, long double angle) final;
  void reflect(const Point& center) final;
  void reflect(const Line& axis) final;
  void move(const Point& vector) final;
  void scale(const Point& center, long double coeff) final;

 protected:
  enum class Moving;
  enum class Scaling;

  std::vector<Point> vertices;
  mutable bool isConvexVal = false;

  const Polygon* toNPolygon(const Shape& other) const;
  static bool compare(const Polygon& first, const Polygon& second,
                      Moving moving, Scaling scaling);
  bool checkSimilar(const Polygon& other, Scaling scaling) const;

  void updIsConvex() const;
  void updArea() const final;
  void updPerimeter() const final;
  void updAll(UpdateCond init) const override;
};

class Ellipse : public Shape {
 public:
  Ellipse();
  Ellipse(const Point& first, const Point& second, long double distance);

  bool operator==(const Ellipse& other) const;
  bool operator==(const Shape& other) const final;

  const std::pair<Point, Point>& focuses() const;
  long double largeAxis() const;
  Point center() const;
  long double lessAxis() const;
  long double eccentricity() const;
  bool isCircle() const;
  std::pair<Line, Line> directrices() const;

  bool isCongruentTo(const Shape& other) const final;
  bool isSimilarTo(const Shape& other) const final;
  bool containsPoint(const Point& point) const final;

  void rotate(const Point& center, long double angle) final;
  void rotateRadians(const Point& center, long double angle) final;
  void reflect(const Point& center) final;
  void reflect(const Line& axis) final;
  void move(const Point& vector) final;
  void scale(const Point& center, long double coeff) final;

 protected:
  std::pair<Point, Point> focusesVal;
  long double largeAxisVal;
  mutable Point centerVal;
  mutable long double lessAxisVal;
  mutable long double eccentricityVal;
  mutable bool isCircleVal = false;
  mutable std::pair<Line, Line> directricesVal;

  void updCenter() const;
  void updLessAxis() const;
  void updEccentricity() const;
  void updIsCircle() const;
  void updDirectrices() const;
  void updArea() const final;
  void updPerimeter() const override;
  void updAll(UpdateCond init) const override;
};

class Circle : public Ellipse {
 public:
  Circle();
  Circle(const Point& center, long double radius);

  long double radius() const;
  std::pair<Line, Line> directrices() const = delete;
  void updDirectrices() const = delete;

 private:
  void updPerimeter() const final;
};

class Triangle : public Polygon {
 public:
  Triangle();
  Triangle(const Point& first, const Point& second, const Point& third);

  Circle inscribedCircle() const;
  Circle circumscribedCircle() const;
  Point centroid() const;
  Point orthocenter() const;
  Line EulerLine() const;
  Circle ninePointsCircle() const;

 private:
  mutable Circle inscribedCircleVal;
  mutable Circle circumscribedCircleVal;
  mutable Point centroidVal;
  mutable Point orthocenterVal;
  mutable Line EulerLineVal;
  mutable Circle ninePointsCircleVal;

  void updInscribedCircle() const;
  void updCircumscribedCircle() const;
  void updCentroid() const;
  void updOrthocenter() const;
  void updEulerLine() const;
  void updNinePointsCircle() const;
  void updAll(UpdateCond init) const final;
};

class Rectangle : public Polygon {
 public:
  Rectangle();
  Rectangle(const Point& first, const Point& second, long double coeff);

  Point center() const;
  std::pair<Line, Line> diagonals();

 protected:
  mutable Point centerVal;
  mutable std::pair<Line, Line> diagonalsVal;

  std::vector<Point> toPolygon(const Point& firstNew, const Point& secondNew,
                               long double coeff);

  void updCenter() const;
  void updDiagonals() const;
  void updAll(UpdateCond init) const override;
};

class Square : public Rectangle {
 public:
  Square();
  Square(const Point& first, const Point& second);

  Circle inscribedCircle();
  Circle circumscribedCircle();

 private:
  mutable Circle inscribedCircleVal;
  mutable Circle circumscribedCircleVal;

  void updInscribedCircle() const;
  void updCircumscribedCircle() const;
  void updAll(UpdateCond init) const final;
};

Point::Point() : x(0), y(0) {}

Point::Point(long double xNew, long double yNew) : x(xNew), y(yNew) {}

Point Point::operator-() const { return Point(-x, -y); }

Point& Point::operator+=(const Point& other) {
  x += other.x;
  y += other.y;
  return *this;
}

Point& Point::operator-=(const Point& other) { return *this += -other; }

Point& Point::operator*=(long double coeff) {
  x *= coeff;
  y *= coeff;
  return *this;
}

Point& Point::operator/=(long double coeff) {
  if (Compare::areEqualLD(coeff, 0)) {
    throw std::runtime_error("Zero divisor.");
  }
  return *this *= 1 / coeff;
}

Point operator+(Point left, const Point& right) { return left += right; }

Point operator-(Point left, const Point& right) { return left -= right; }

Point operator*(Point point, long double coeff) { return point *= coeff; }

Point operator*(long double coeff, Point point) { return point *= coeff; }

Point operator/(Point point, long double coeff) { return point /= coeff; }

bool Point::operator==(const Point& other) const {
  Point shrinked = *this - other;
  return Compare::areEqualLD(shrinked.x, 0) &&
         Compare::areEqualLD(shrinked.y, 0);
}

long double Point::distance(const Point& other) const {
  return std::hypot(x - other.x, y - other.y);
}

long double Point::scalarMultiply(const Point& other) const {
  return x * other.x + y * other.y;
}

long double Point::vectorMultiply(const Point& other) const {
  return x * other.y - y * other.x;
}

long double Point::absolute() const { return distance(Point(0, 0)); }

Point Point::toUnit() const { return *this / absolute(); }

Point Point::toPositiveX() const { return (x < 0) ? -*this : *this; }

bool Point::isCollinearTo(const Point& other) const {
  return Compare::areEqualLD(vectorMultiply(other), 0);
}

Point Point::rotate90() const { return Point(-y, x); }

void Point::rotate(const Point& center, long double angle) {
  rotateRadians(center, Angle::toRadians(angle));
}

void Point::rotateRadians(const Point& center, long double angle) {
  Point byCenter = *this - center;
  Point rotated(byCenter.x * std::cos(angle) - byCenter.y * std::sin(angle),
                byCenter.x * std::sin(angle) + byCenter.y * std::cos(angle));
  *this = rotated + center;
}

void Point::reflect(const Point& center) { *this = 2 * center - *this; }

void Point::reflect(const Line& axis) {
  long double coeff = scalarMultiply(axis.getNormal()) - axis.getDistance();
  *this = *this - 2 * coeff * axis.getNormal();
}

void Point::scale(const Point& center, long double coeff) {
  *this = center + coeff * (*this - center);
}

std::istream& operator>>(std::istream& inp, Point& point) {
  inp >> point.x >> point.y;
  return inp;
}

std::ostream& operator<<(std::ostream& out, const Point& point) {
  return out << point.x << ' ' << point.y << ' ';
}

Line::Line() : normal(1, 0), distance(1) {}

Line::Line(const Point& first, const Point& second)
    : normal(Point(second - first).rotate90().toUnit().toPositiveX()),
      distance(first.scalarMultiply(normal)) {}

Line::Line(long double coeff, long double shift)
    : normal(Point(1, coeff).rotate90().toUnit().toPositiveX()),
      distance(Point(0, shift).scalarMultiply(normal)) {}

Line::Line(const Point& point, long double coeff)
    : normal(Point(1, coeff).rotate90().toUnit().toPositiveX()),
      distance(point.scalarMultiply(normal)) {}

bool Line::operator==(const Line& other) const {
  return normal == other.normal &&
         Compare::areEqualLD(distance, other.distance);
}

Point Line::getNormal() const { return normal; }

long double Line::getDistance() const { return distance; }

enum class Shape::UpdateCond { NotInitialized, Initialized };

Shape::Shape()
    : areaVal(0), perimeterVal(0), initialized(UpdateCond::NotInitialized) {}

long double Shape::area() const {
  updAll(UpdateCond::NotInitialized);
  return areaVal;
}

long double Shape::perimeter() const {
  updAll(UpdateCond::NotInitialized);
  return perimeterVal;
}

void Shape::updAll(UpdateCond condition) const {
  if (initialized == condition) {
    updArea();
    updPerimeter();
    initialized = UpdateCond::Initialized;
  }
}

enum class Polygon::Moving { No, Yes };
enum class Polygon::Scaling { No, Yes };

Polygon::Polygon()
    : Shape(), vertices({Point(0, 0), Point(1, 0), Point(0, 1)}) {}

Polygon::Polygon(const Point& first, const Point& second, const Point& third)
    : Shape(), vertices({first, second, third}) {}

Polygon::Polygon(const std::vector<Point>& points) : Shape(), vertices(points) {
  if (vertices.size() < 3) {
    throw std::invalid_argument("Too few points.");
  }
}

template <typename... Args>
Polygon::Polygon(Args... points) : Shape() {
  static_assert((std::is_same_v<Point, Args> && ...));
  static_assert(sizeof...(points) >= 3);
  (..., vertices.push_back(points));
}

bool Polygon::operator==(const Shape& other) const {
  const Polygon* otherCasted = toNPolygon(other);
  return otherCasted ? *this == *otherCasted : false;
}

bool Polygon::operator==(const Polygon& other) const {
  Polygon copy = *this;
  if (compare(other, copy, Moving::No, Scaling::No)) {
    return true;
  }
  std::reverse(copy.vertices.begin(), copy.vertices.end());
  return compare(other, copy, Moving::No, Scaling::No);
}

size_t Polygon::verticesCount() const { return vertices.size(); }

const std::vector<Point>& Polygon::getVertices() const { return vertices; }

bool Polygon::isConvex() const {
  updAll(UpdateCond::NotInitialized);
  return isConvexVal;
}

bool Polygon::isCongruentTo(const Shape& other) const {
  const Polygon* otherCasted = toNPolygon(other);
  return otherCasted ? checkSimilar(*otherCasted, Scaling::No) : false;
}

bool Polygon::isSimilarTo(const Shape& other) const {
  const Polygon* otherCasted = toNPolygon(other);
  return otherCasted ? checkSimilar(*otherCasted, Scaling::Yes) : false;
}

bool Polygon::containsPoint(const Point& point) const {
  bool contains = false;
  for (size_t index = 0; index < vertices.size(); ++index) {
    size_t next = (index + 1) % vertices.size();
    long double pointCurrent = point.vectorMultiply(vertices[index]);
    long double pointNext = point.vectorMultiply(vertices[next]);
    long double currentNext = vertices[index].vectorMultiply(vertices[next]);
    long double sideValue = (pointCurrent - pointNext + currentNext) /
                            (vertices[index].y - vertices[next].y);
    bool isIntersect =
        (vertices[index].y - point.y < 0) != (vertices[next].y - point.y < 0);

    if (isIntersect && sideValue < 0) {
      contains = !contains;
    }
  }
  return contains;
}

void Polygon::rotate(const Point& center, long double angle) {
  for (auto& point : vertices) {
    point.rotate(center, angle);
  }
}

void Polygon::rotateRadians(const Point& center, long double angle) {
  for (auto& point : vertices) {
    point.rotateRadians(center, angle);
  }
}

void Polygon::reflect(const Point& center) {
  for (auto& point : vertices) {
    point.reflect(center);
  }
}

void Polygon::reflect(const Line& axis) {
  for (auto& point : vertices) {
    point.reflect(axis);
  }
}

void Polygon::move(const Point& vector) {
  for (auto& point : vertices) {
    point += vector;
  }
}

void Polygon::scale(const Point& center, long double coeff) {
  for (auto& point : vertices) {
    point.scale(center, coeff);
  }
  updAll(UpdateCond::Initialized);
}

const Polygon* Polygon::toNPolygon(const Shape& other) const {
  const Polygon* otherCasted = dynamic_cast<const Polygon*>(&other);
  return otherCasted != nullptr &&
                 vertices.size() == otherCasted->vertices.size()
             ? otherCasted
             : nullptr;
}

bool Polygon::compare(const Polygon& first, const Polygon& second,
                      Moving moving, Scaling scaling) {
  Polygon copy = second;
  Point verticePinned = first.vertices[0];
  Point sidePinned = first.vertices[1] - verticePinned;
  long double anglePinned = std::atan2(sidePinned.y, sidePinned.x);
  for (size_t index = 0; index < copy.vertices.size(); ++index) {
    copy = second;
    size_t next = (index + 1) % copy.vertices.size();
    Point sideCheck = copy.vertices[next] - copy.vertices[index];

    if (moving == Moving::Yes) {
      long double angle = anglePinned - std::atan2(sideCheck.y, sideCheck.x);
      copy.rotateRadians(verticePinned, angle);
      Point shift = verticePinned - copy.vertices[index];
      copy.move(shift);
    }

    if (scaling == Scaling::Yes) {
      long double coeff = sidePinned.absolute() / sideCheck.absolute();
      copy.scale(verticePinned, coeff);
    }

    bool similar = true;
    for (size_t vertice = 0; vertice < copy.vertices.size(); ++vertice) {
      size_t indexRotated = (vertice + index) % copy.vertices.size();
      if (copy.vertices[indexRotated] != first.vertices[vertice]) {
        similar = false;
        break;
      }
    }

    if (similar) {
      return true;
    }
  }
  return false;
}

bool Polygon::checkSimilar(const Polygon& other, Scaling scaling) const {
  Polygon copy;
  for (size_t iteration = 0; iteration < 4; ++iteration) {
    copy = *this;
    if (iteration & 1) {
      std::reverse(copy.vertices.begin(), copy.vertices.end());
    }
    if (iteration & 2) {
      copy.reflect(Line(Point(0, 0), Point(1, 1)));
    }
    if (compare(other, copy, Moving::Yes, scaling)) {
      return true;
    }
  }
  return false;
}

void Polygon::updIsConvex() const {
  Point firstSegment(vertices[0] - vertices.back());
  Point secondSegment(vertices[1] - vertices[0]);
  bool orientation = firstSegment.vectorMultiply(secondSegment) > 0;

  for (size_t index = 1; index < vertices.size(); ++index) {
    size_t next = (index + 1) % vertices.size();
    firstSegment = secondSegment;
    secondSegment = vertices[next] - vertices[index];

    if (orientation != (firstSegment.vectorMultiply(secondSegment) > 0)) {
      isConvexVal = false;
      return;
    }
  }
  isConvexVal = true;
}

void Polygon::updArea() const {
  areaVal = 0;
  for (size_t index = 0; index < vertices.size(); ++index) {
    size_t next = (index + 1) % vertices.size();
    long double height = vertices[index].x - vertices[next].x;
    long double halfSumSides = (vertices[index].y + vertices[next].y) / 2;
    areaVal += height * (halfSumSides - vertices[0].y);
  }
  areaVal = std::abs(areaVal);
}

void Polygon::updPerimeter() const {
  perimeterVal = 0;
  for (size_t index = 0; index < vertices.size(); ++index) {
    size_t next = (index + 1) % vertices.size();
    perimeterVal += vertices[index].distance(vertices[next]);
  }
}

void Polygon::updAll(UpdateCond condition) const {
  if (initialized == condition) {
    updIsConvex();
    updArea();
    updPerimeter();
    initialized = UpdateCond::Initialized;
  }
}

Ellipse::Ellipse()
    : Shape(),
      focusesVal(std::pair<Point, Point>(Point(0, 0), Point(0, 0))),
      largeAxisVal(1) {}

Ellipse::Ellipse(const Point& first, const Point& second, long double distance)
    : Shape(),
      focusesVal(std::pair<Point, Point>(first, second)),
      largeAxisVal(distance / 2) {
  if (distance < first.distance(second)) {
    throw std::runtime_error("Invalid ellipse.");
  }
}

bool Ellipse::operator==(const Ellipse& other) const {
  return Compare::areEqualPair(focusesVal, other.focusesVal) &&
         Compare::areEqualLD(largeAxisVal, other.largeAxisVal);
}

bool Ellipse::operator==(const Shape& other) const {
  const Ellipse* otherCasted = dynamic_cast<const Ellipse*>(&other);
  return otherCasted ? *this == *otherCasted : false;
}

const std::pair<Point, Point>& Ellipse::focuses() const { return focusesVal; }

long double Ellipse::largeAxis() const { return largeAxisVal; }

Point Ellipse::center() const {
  updAll(UpdateCond::NotInitialized);
  return centerVal;
}

long double Ellipse::lessAxis() const {
  updAll(UpdateCond::NotInitialized);
  return lessAxisVal;
}

long double Ellipse::eccentricity() const {
  updAll(UpdateCond::NotInitialized);
  return eccentricityVal;
}

bool Ellipse::isCircle() const {
  updAll(UpdateCond::NotInitialized);
  return isCircleVal;
}

std::pair<Line, Line> Ellipse::directrices() const {
  updAll(UpdateCond::NotInitialized);
  return directricesVal;
}

bool Ellipse::isCongruentTo(const Shape& other) const {
  const Ellipse* otherCasted = dynamic_cast<const Ellipse*>(&other);
  if (otherCasted == nullptr) {
    return false;
  }
  return Compare::areEqualLD(largeAxisVal, otherCasted->largeAxisVal) &&
         Compare::areEqualLD(eccentricityVal, otherCasted->eccentricityVal);
}

bool Ellipse::isSimilarTo(const Shape& other) const {
  const Ellipse* otherCasted = dynamic_cast<const Ellipse*>(&other);
  if (otherCasted == nullptr) {
    return false;
  }
  return Compare::areEqualLD(eccentricityVal, otherCasted->eccentricityVal);
}

bool Ellipse::containsPoint(const Point& point) const {
  return focusesVal.first.distance(point) + focusesVal.second.distance(point) <
         2 * largeAxisVal;
}

void Ellipse::rotate(const Point& center, long double angle) {
  focusesVal.first.rotate(center, angle);
  focusesVal.second.rotate(center, angle);
  updAll(UpdateCond::Initialized);
}

void Ellipse::rotateRadians(const Point& center, long double angle) {
  focusesVal.first.rotateRadians(center, angle);
  focusesVal.second.rotateRadians(center, angle);
  updAll(UpdateCond::Initialized);
}

void Ellipse::reflect(const Point& center) {
  focusesVal.first.reflect(center);
  focusesVal.second.reflect(center);
  updAll(UpdateCond::Initialized);
}

void Ellipse::reflect(const Line& axis) {
  focusesVal.first.reflect(axis);
  focusesVal.second.reflect(axis);
  updAll(UpdateCond::Initialized);
}

void Ellipse::move(const Point& vector) {
  focusesVal.first += vector;
  focusesVal.second += vector;
  updAll(UpdateCond::Initialized);
}

void Ellipse::scale(const Point& center, long double coeff) {
  focusesVal.first.scale(center, coeff);
  focusesVal.second.scale(center, coeff);
  largeAxisVal *= coeff;
  updAll(UpdateCond::Initialized);
}

void Ellipse::updCenter() const {
  centerVal = (focusesVal.first + focusesVal.second) / 2;
}

void Ellipse::updLessAxis() const {
  long double focusDistance = centerVal.distance(focusesVal.first);
  lessAxisVal =
      std::sqrt(largeAxisVal * largeAxisVal - focusDistance * focusDistance);
}

void Ellipse::updEccentricity() const {
  eccentricityVal =
      focusesVal.first.distance(focusesVal.second) / (2 * largeAxisVal);
}

void Ellipse::updIsCircle() const {
  isCircleVal = focusesVal.first == focusesVal.second;
}

void Ellipse::updDirectrices() const {
  if (Compare::areEqualLD(eccentricityVal, 0)) {
    return;
  }

  Point focusVector = focusesVal.second - focusesVal.first;
  Point shift = focusVector.toUnit() * largeAxisVal / eccentricityVal;
  Line first(centerVal + shift, centerVal + shift + focusVector.rotate90());
  Line second(centerVal - shift, centerVal - shift + focusVector.rotate90());
  directricesVal = std::pair<Line, Line>(first, second);
}

void Ellipse::updArea() const { areaVal = M_PIl * largeAxisVal * lessAxisVal; }

void Ellipse::updPerimeter() const {
  perimeterVal = 4 * largeAxisVal * std::comp_ellint_2(eccentricityVal);
}

void Ellipse::updAll(UpdateCond condition) const {
  if (initialized == condition) {
    updCenter();
    updLessAxis();
    updEccentricity();
    updIsCircle();
    updDirectrices();
    updArea();
    updPerimeter();
    initialized = UpdateCond::Initialized;
  }
}

Circle::Circle() : Ellipse() {}

Circle::Circle(const Point& center, long double radius)
    : Ellipse(center, center, 2 * radius) {}

long double Circle::radius() const { return largeAxisVal; }

void Circle::updPerimeter() const { perimeterVal = 2 * M_PIl * largeAxisVal; }

Triangle::Triangle() : Polygon() {}

Triangle::Triangle(const Point& first, const Point& second, const Point& third)
    : Polygon(first, second, third) {}

Circle Triangle::inscribedCircle() const {
  updAll(UpdateCond::NotInitialized);
  return inscribedCircleVal;
}

Circle Triangle::circumscribedCircle() const {
  updAll(UpdateCond::NotInitialized);
  return circumscribedCircleVal;
}

Point Triangle::centroid() const {
  updAll(UpdateCond::NotInitialized);
  return centroidVal;
}

Point Triangle::orthocenter() const {
  updAll(UpdateCond::NotInitialized);
  return orthocenterVal;
}

Line Triangle::EulerLine() const {
  updAll(UpdateCond::NotInitialized);
  return EulerLineVal;
}

Circle Triangle::ninePointsCircle() const {
  updAll(UpdateCond::NotInitialized);
  return ninePointsCircleVal;
}

void Triangle::updInscribedCircle() const {
  std::vector<long double> length(3);
  for (size_t index = 0; index < 3; ++index) {
    length[index] =
        vertices[(index + 1) % 3].distance(vertices[(index + 2) % 3]);
  }
  Point center;
  for (size_t index = 0; index < 3; ++index) {
    center.x += vertices[index].x * length[index];
    center.y += vertices[index].y * length[index];
  }
  center /= (length[0] + length[1] + length[2]);
  inscribedCircleVal = Circle(center, 2 * areaVal / perimeterVal);
}

void Triangle::updCircumscribedCircle() const {
  std::vector<Point> diff(3);
  std::vector<long double> mods(3);
  for (size_t index = 0; index < 3; ++index) {
    diff[index] = vertices[index] - vertices[(index + 1) % 3];
    mods[index] = vertices[index].scalarMultiply(vertices[index]);
  }
  Point coeff;
  for (size_t index = 0; index < 3; ++index) {
    coeff.x += diff[index].x * mods[(index + 2) % 3];
    coeff.y += diff[index].y * mods[(index + 2) % 3];
  }
  Point center = coeff.rotate90() / (2 * diff[0].vectorMultiply(diff[2]));
  circumscribedCircleVal = Circle(center, center.distance(vertices[0]));
}

void Triangle::updCentroid() const {
  centroidVal = (vertices[0] + vertices[1] + vertices[2]) / 3;
}

void Triangle::updOrthocenter() const {
  orthocenterVal = 3 * centroidVal - 2 * circumscribedCircleVal.center();
}

void Triangle::updEulerLine() const {
  EulerLineVal = Line(centroidVal, orthocenterVal);
}

void Triangle::updNinePointsCircle() const {
  Circle circle = circumscribedCircleVal;
  ninePointsCircleVal =
      Circle((circle.center() + orthocenterVal) / 2, circle.radius() / 2);
}

void Triangle::updAll(UpdateCond condition) const {
  if (initialized == condition) {
    updArea();
    updPerimeter();
    updInscribedCircle();
    updCircumscribedCircle();
    updCentroid();
    updOrthocenter();
    updEulerLine();
    updNinePointsCircle();
    initialized = UpdateCond::Initialized;
  }
}

Rectangle::Rectangle()
    : Polygon(Point(0, 0), Point(1, 0), Point(0, 1), Point(1, 1)) {}

Rectangle::Rectangle(const Point& first, const Point& second, long double coeff)
    : Polygon(toPolygon(first, second, coeff)) {}

Point Rectangle::center() const {
  updAll(UpdateCond::NotInitialized);
  return centerVal;
}

std::pair<Line, Line> Rectangle::diagonals() {
  updAll(UpdateCond::NotInitialized);
  return diagonalsVal;
}

std::vector<Point> Rectangle::toPolygon(const Point& first, const Point& second,
                                        long double coeff) {
  if (Compare::areEqualLD(coeff, 0)) {
    throw std::runtime_error("Invalid coefficient");
  }

  long double powCoeff = coeff * coeff;
  long double length = coeff * first.distance(second) / (1 + powCoeff);
  Point vector = (second - first).toUnit();
  Point shift = vector * std::min(powCoeff, 1.0L) / (1 + powCoeff);

  Point third = first + shift + length * vector.rotate90();
  Point fourth = second - shift - length * vector.rotate90();
  return {first, third, second, fourth};
}

void Rectangle::updCenter() const {
  centerVal = (vertices[0] + vertices[2]) / 2;
}

void Rectangle::updDiagonals() const {
  diagonalsVal = std::pair<Line, Line>(Line(vertices[0], vertices[2]),
                                       Line(vertices[1], vertices[3]));
}

void Rectangle::updAll(UpdateCond condition) const {
  if (initialized == condition) {
    updCenter();
    updDiagonals();
    updArea();
    updPerimeter();
    initialized = UpdateCond::Initialized;
  }
}

Square::Square() : Rectangle() {}

Square::Square(const Point& first, const Point& second)
    : Rectangle(first, second, 1) {}

Circle Square::inscribedCircle() {
  updAll(UpdateCond::NotInitialized);
  return inscribedCircleVal;
}

Circle Square::circumscribedCircle() {
  updAll(UpdateCond::NotInitialized);
  return circumscribedCircleVal;
}

void Square::updInscribedCircle() const {
  inscribedCircleVal = Circle(centerVal, vertices[0].distance(vertices[1]) / 2);
}

void Square::updCircumscribedCircle() const {
  circumscribedCircleVal =
      Circle(centerVal, vertices[0].distance(vertices[1]) / std::sqrt(2));
}

void Square::updAll(UpdateCond condition) const {
  if (initialized == condition) {
    updCenter();
    updDiagonals();
    updInscribedCircle();
    updCircumscribedCircle();
    updArea();
    updPerimeter();
    initialized = UpdateCond::Initialized;
  }
}
