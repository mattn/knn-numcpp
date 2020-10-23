#include <NumCpp.hpp>
#include <vector>
#include <map>
#include <string>
#include <iostream>

typedef struct {
  int k;
  nc::NdArray<float> XX;
  std::vector<std::string> Y;
} KNN;

static float
distance(nc::NdArray<float> lhs, nc::NdArray<float> rhs) {
  float val = 0;
  for (uint32_t i = 0; i < lhs.size(); i++)
    val += pow(lhs[i]-rhs[i], 2);
  return sqrt(val);
}

typedef struct {
  uint32_t i;
  float f;
} item;

typedef struct {
  int i;
  std::string s;
} rank;

static std::vector<std::string>
predict(KNN& knn, nc::NdArray<float>& X) {
  std::vector<std::string> results;
  for (uint32_t n = 0; n < X.numRows(); n++) {
    auto&& x = X.row(n);
    std::vector<item> items;
    for (uint32_t i = 0; i < knn.XX.numRows(); i++) {
      auto&& xx = knn.XX.row(i);
      items.push_back({
        .i =  i,
        .f = distance(x, xx),
      });
    }
    std::sort(items.begin(), items.end(), [](auto const& lhs, auto const& rhs) {
      return lhs.f < rhs.f;
    });
    std::vector<std::string> labels;
    for (int i = 0; i < knn.k; i++) {
      labels.push_back(knn.Y[items[i].i]);
    }
    std::map<std::string, int> founds;
    for (auto& label : labels) {
      founds[label] = 1;
    }

    std::vector<rank> ranks;
    for (auto& it : founds) {
      ranks.push_back({
        .i = it.second,
        .s = it.first,
      });
    }

    std::sort(ranks.begin(), ranks.end(), [](auto const& lhs, auto const& rhs) {
      return lhs.i > rhs.i;
    });
    results.push_back(ranks[0].s);
  }
  return results;
}


static std::vector<std::string>
split(std::string& filename, char delimiter) {
  std::istringstream f(filename);
  std::string field;
  std::vector<std::string> result;
  while (getline(f, field, delimiter)) {
    result.push_back(field);
  }
  return result;
}

int
main() {
  std::ifstream ifs("iris.csv");

  std::string line;

  // skip header
  std::getline(ifs, line);

  std::vector<float> rows;
  std::vector<std::string> names;
  while (std::getline(ifs, line)) {
    // sepal length, sepal width, petal length, petal width, name
    auto cells = split(line, ',');
    rows.push_back(std::stof(cells.at(0)));
    rows.push_back(std::stof(cells.at(1)));
    rows.push_back(std::stof(cells.at(2)));
    rows.push_back(std::stof(cells.at(3)));
    names.push_back(cells.at(4));
  }
  ifs.close();

  // make vector 4 dimensions
  nc::NdArray<float> X(rows);
  X.reshape((uint32_t) rows.size()/4, 4);

  // make factor from input values
  KNN knn = {
    .k = 8,
    .XX = X,
    .Y = names,
  };
  const auto predicted = predict(knn, X);

  // predict samples
  size_t count = 0;
  for (size_t n = 0; n < predicted.size(); n++) {
    if (predicted[n] == names[n]) {
      count++;
    }
  }

  std::cout << ((float)count / (float)predicted.size()) << std::endl;
  return 0;
}

