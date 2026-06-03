#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <cmath>

using namespace cv;
using namespace std;

/////////////////////////////
// Constants
/////////////////////////////

const int BKG_THRESH = 60;

const int CORNER_WIDTH = 32;
const int CORNER_HEIGHT = 84;

const int RANK_WIDTH = 70;
const int RANK_HEIGHT = 125;

const int SUIT_WIDTH = 70;
const int SUIT_HEIGHT = 100;

const double RANK_HOG_MAX = 1.00;
const double SUIT_HOG_MAX = 1.20;

const int CARD_MAX_AREA = 1000000;
const int CARD_MIN_AREA = 1000;

/////////////////////////////
// Structures
/////////////////////////////

struct TrainHOG {
    string        name;
    vector<float> descriptor; // pre-computed HOG
};

struct QueryCard {
    vector<Point>   contour;
    vector<Point2f> corner_pts;

    int   width = 0;
    int   height = 0;
    Point center;

    Mat warp;
    Mat rank_img;
    Mat suit_img;

    string best_rank_match = "Unknown";
    string best_suit_match = "Unknown";

    double rank_dist = 0.0;
    double suit_dist = 0.0;
};

/////////////////////////////
// HOG helpers
/////////////////////////////

static vector<float> computeHOG(const Mat& grayImg,
    int cellsX = 7,
    int cellsY = 7,
    int nBins = 9)
{
    // Resize to a canonical size so descriptors are comparable
    Mat img;
    resize(grayImg, img, Size(RANK_WIDTH, RANK_HEIGHT));

    // Compute gradients
    Mat gx, gy;
    Sobel(img, gx, CV_32F, 1, 0, 1);
    Sobel(img, gy, CV_32F, 0, 1, 1);

    Mat mag, angle;
    cartToPolar(gx, gy, mag, angle, true);

    int cellW = img.cols / cellsX;
    int cellH = img.rows / cellsY;

    vector<float> descriptor;
    descriptor.reserve(cellsX * cellsY * nBins);

    for (int cy = 0; cy < cellsY; ++cy) {
        for (int cx = 0; cx < cellsX; ++cx) {
            int x0 = cx * cellW;
            int y0 = cy * cellH;
            int x1 = min(x0 + cellW, img.cols);
            int y1 = min(y0 + cellH, img.rows);

            Rect roi(x0, y0, x1 - x0, y1 - y0);
            Mat cellMag = mag(roi);
            Mat cellAngle = angle(roi);

            vector<float> hist(nBins, 0.f);

            for (int r = 0; r < cellMag.rows; ++r) {
                const float* mp = cellMag.ptr<float>(r);
                const float* ap = cellAngle.ptr<float>(r);
                for (int c = 0; c < cellMag.cols; ++c) {
                    float a = fmod(ap[c], 180.f);
                    int   bin = static_cast<int>(a / 180.f * static_cast<float>(nBins)) % nBins;
                    hist[bin] += mp[c];
                }
            }
            for (float v : hist) descriptor.push_back(v);
        }
    }

    float norm = 0.f;
    for (float v : descriptor) norm += v * v;
    norm = sqrtf(norm) + 1e-6f;
    for (float& v : descriptor) v /= norm;

    return descriptor;
}

static double hogDistance(const vector<float>& a, const vector<float>& b)
{
    double d = 0.0;
    size_t len = min(a.size(), b.size());
    for (size_t i = 0; i < len; ++i) {
        double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        d += diff * diff;
    }
    return sqrt(d);
}

/////////////////////////////
// Helpers
/////////////////////////////

static vector<Point2f> orderPoints(const vector<Point2f>& pts)
{
    vector<Point2f> rect(4);
    float sumMin = 1e9f, sumMax = -1e9f;
    float diffMin = 1e9f, diffMax = -1e9f;

    for (const auto& p : pts) {
        float s = p.x + p.y;
        float d = p.y - p.x;
        if (s < sumMin) { sumMin = s; rect[0] = p; }
        if (s > sumMax) { sumMax = s; rect[2] = p; }
        if (d < diffMin) { diffMin = d; rect[1] = p; }
        if (d > diffMax) { diffMax = d; rect[3] = p; }
    }
    return rect;
}

/////////////////////////////
// Load & pre-compute HOG
/////////////////////////////

static vector<TrainHOG> loadRankHOGs(const string& path)
{
    const vector<string> names = {
        "Ace","Two","Three","Four","Five","Six","Seven",
        "Eight","Nine","Ten","Jack","Queen","King"
    };

    vector<TrainHOG> ranks;
    for (const auto& name : names) {
        Mat img = imread(path + "/" + name + ".jpg", IMREAD_GRAYSCALE);
        if (img.empty()) {
            cout << "Missing rank: " << name << endl;
            continue;
        }

        Mat thr;
        threshold(img, thr, 0, 255, THRESH_BINARY | THRESH_OTSU);

        TrainHOG t;
        t.name = name;
        t.descriptor = computeHOG(thr);
        ranks.push_back(t);
    }
    return ranks;
}

static vector<TrainHOG> loadSuitHOGs(const string& path)
{
    const vector<string> names = { "Spades","Diamonds","Clubs","Hearts" };

    vector<TrainHOG> suits;
    for (const auto& name : names) {
        Mat img = imread(path + "/" + name + ".jpg", IMREAD_GRAYSCALE);
        if (img.empty()) {
            cout << "Missing suit: " << name << endl;
            continue;
        }

        Mat thr;
        threshold(img, thr, 0, 255, THRESH_BINARY | THRESH_OTSU);

        TrainHOG t;
        t.name = name;
        t.descriptor = computeHOG(thr);
        suits.push_back(t);
    }
    return suits;
}

/////////////////////////////
// Preprocess Image
/////////////////////////////

static Mat preprocessImage(const Mat& image)
{
    Mat gray, blurImg, thresh;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, blurImg, Size(5, 5), 0);

    int bkg_level = static_cast<int>(gray.at<uchar>(gray.rows / 100, gray.cols / 2));
    int thresh_level = bkg_level + BKG_THRESH;

    threshold(blurImg, thresh, thresh_level, 255, THRESH_BINARY);
    return thresh;
}

/////////////////////////////
// Find Cards
/////////////////////////////

static vector<vector<Point>> findCards(const Mat& thresh)
{
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(thresh, contours, hierarchy,
        RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    vector<vector<Point>> cards;
    for (const auto& contour : contours) {
        double area = contourArea(contour);
        if (area < static_cast<double>(CARD_MIN_AREA) ||
            area > static_cast<double>(CARD_MAX_AREA))
            continue;

        vector<Point> approx;
        approxPolyDP(contour, approx,
            arcLength(contour, true) * 0.02, true);

        if (approx.size() == 4 && isContourConvex(approx))
            cards.push_back(approx);
    }
    return cards;
}

/////////////////////////////
// Flatten Card
/////////////////////////////

static Mat flattener(const Mat& image, const vector<Point2f>& pts,
    int /*w*/, int /*h*/)
{
    vector<Point2f> rect = orderPoints(pts);
    vector<Point2f> dst = {
        {0.f,   0.f},
        {199.f, 0.f},
        {199.f, 299.f},
        {0.f,   299.f}
    };

    Mat M = getPerspectiveTransform(rect, dst);
    Mat warp;
    warpPerspective(image, warp, M, Size(200, 300));
    cvtColor(warp, warp, COLOR_BGR2GRAY);
    return warp;
}

/////////////////////////////
// Preprocess Card
/////////////////////////////

static QueryCard preprocessCard(const vector<Point>& contour, const Mat& image)
{
    QueryCard qCard;
    qCard.contour = contour;

    Rect box = boundingRect(contour);

    vector<Point2f> pts;
    Point2f center(0.f, 0.f);
    for (const auto& p : contour) {
        center.x += static_cast<float>(p.x);
        center.y += static_cast<float>(p.y);
        pts.emplace_back(static_cast<float>(p.x), static_cast<float>(p.y));
    }
    float n = static_cast<float>(contour.size());
    center.x /= n;
    center.y /= n;
    qCard.center = Point(center);

    qCard.warp = flattener(image, pts, box.width, box.height);

    Mat warpThresh;
    threshold(qCard.warp, warpThresh, 0, 255,
        THRESH_BINARY_INV | THRESH_OTSU);

    Mat corner = warpThresh(Rect(0, 0, CORNER_WIDTH, CORNER_HEIGHT)).clone();
    int split = static_cast<int>(corner.rows * 0.6f);

    Mat rank = corner(Rect(0, 0, corner.cols, split)).clone();
    Mat suit = corner(Rect(0, split, corner.cols, corner.rows - split)).clone();

    // Rank ROI
    {
        Mat tmp = rank.clone();
        vector<vector<Point>> cnts;
        findContours(tmp, cnts, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        if (!cnts.empty()) {
            sort(cnts.begin(), cnts.end(),
                [](const vector<Point>& a, const vector<Point>& b) {
                    return contourArea(a) > contourArea(b); });
            Rect r = boundingRect(cnts[0]);
            r &= Rect(0, 0, rank.cols, rank.rows);
            if (r.area() > 0)
                resize(rank(r), qCard.rank_img, Size(RANK_WIDTH, RANK_HEIGHT));
        }
    }

    // Suit ROI
    {
        Mat tmp = suit.clone();
        vector<vector<Point>> cnts;
        findContours(tmp, cnts, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        if (!cnts.empty()) {
            sort(cnts.begin(), cnts.end(),
                [](const vector<Point>& a, const vector<Point>& b) {
                    return contourArea(a) > contourArea(b); });
            Rect r = boundingRect(cnts[0]);
            r &= Rect(0, 0, suit.cols, suit.rows);
            if (r.area() > 0)
                resize(suit(r), qCard.suit_img, Size(SUIT_WIDTH, SUIT_HEIGHT));
        }
    }

    return qCard;
}

/////////////////////////////
// Match Card  (HOG-based)
/////////////////////////////

static void matchCard(QueryCard& qCard,
    const vector<TrainHOG>& ranks,
    const vector<TrainHOG>& suits)
{
    double best_rank_dist = 1e9;
    double best_suit_dist = 1e9;

    // Rank matching
    if (!qCard.rank_img.empty()) {
        const vector<float> queryRankHOG = computeHOG(qCard.rank_img);
        for (const auto& r : ranks) {
            double d = hogDistance(queryRankHOG, r.descriptor);
            if (d < best_rank_dist) {
                best_rank_dist = d;
                qCard.best_rank_match = r.name;
            }
        }
    }

    // Suit matching
    if (!qCard.suit_img.empty()) {
        const vector<float> querySuitHOG = computeHOG(qCard.suit_img);
        for (const auto& s : suits) {
            double d = hogDistance(querySuitHOG, s.descriptor);
            if (d < best_suit_dist) {
                best_suit_dist = d;
                qCard.best_suit_match = s.name;
            }
        }
    }

    // Reject weak matches
    if (best_rank_dist > RANK_HOG_MAX) qCard.best_rank_match = "Unknown";
    if (best_suit_dist > SUIT_HOG_MAX) qCard.best_suit_match = "Unknown";

    qCard.rank_dist = best_rank_dist;
    qCard.suit_dist = best_suit_dist;
}

/////////////////////////////
// Draw Results
/////////////////////////////

static void drawResults(Mat& image, const QueryCard& qCard)
{
    circle(image, qCard.center, 5, Scalar(255, 0, 0), FILLED);

    const string text = qCard.best_rank_match + " of " + qCard.best_suit_match;

    const double fontScale = std::clamp(
        static_cast<double>(image.cols) / 1000.0, 0.3, 1.0);

    putText(image, text,
        Point(qCard.center.x - 60, qCard.center.y),
        FONT_HERSHEY_SIMPLEX, fontScale,
        Scalar(0, 255, 0),
        std::max(1, static_cast<int>(fontScale * 2.0)));
}

/////////////////////////////
// Main
/////////////////////////////

int main(int argc, char* argv[])
{
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " <image_file>" << endl;
        return -1;
    }

    const string imagePath = argv[1];

    Mat image = imread(imagePath);
    if (image.empty()) {
        cout << "Could not open image: " << imagePath << endl;
        return -1;
    }

    // Load training images and pre-compute HOG descriptors once
    const vector<TrainHOG> trainRanks = loadRankHOGs("Card_Imgs");
    const vector<TrainHOG> trainSuits = loadSuitHOGs("Card_Imgs");

    const Mat thresh = preprocessImage(image);

    const vector<vector<Point>> cards = findCards(thresh);
    cout << "Cards found: " << cards.size() << endl;

    for (const auto& c : cards) {
        QueryCard qCard = preprocessCard(c, image);
        matchCard(qCard, trainRanks, trainSuits);
        drawResults(image, qCard);

        cout << qCard.best_rank_match << " of "
             << qCard.best_suit_match
             << "  (rank_dist=" << qCard.rank_dist
             << "  suit_dist=" << qCard.suit_dist
             << ")" << endl;
    }

    imshow("Detected Cards", image);
    waitKey(0);

    return 0;
}
