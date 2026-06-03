#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

using namespace cv;
using namespace std;

/////////////////////////////
// Constants
/////////////////////////////

const int BKG_THRESH = 60;
const int CARD_THRESH = 30;

const int CORNER_WIDTH = 32;
const int CORNER_HEIGHT = 84;

const int RANK_WIDTH = 70;
const int RANK_HEIGHT = 125;

const int SUIT_WIDTH = 70;
const int SUIT_HEIGHT = 100;

const int RANK_DIFF_MAX = 3000;
const int SUIT_DIFF_MAX = 900;

const int CARD_MAX_AREA = 1000000;
const int CARD_MIN_AREA = 1000;

/////////////////////////////
// Structures
/////////////////////////////

struct TrainImage {
    string name;
    Mat img;
};

struct QueryCard {
    vector<Point> contour;
    vector<Point2f> corner_pts;

    int width = 0;
    int height = 0;

    Point center;

    Mat warp;
    Mat rank_img;
    Mat suit_img;

    string best_rank_match = "Unknown";
    string best_suit_match = "Unknown";

    int rank_diff = 0;
    int suit_diff = 0;
};

/////////////////////////////
// Helpers
/////////////////////////////

vector<Point2f> orderPoints(const vector<Point2f>& pts) {
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
// Load Training Images
/////////////////////////////

vector<TrainImage> loadRanks(const string& path) {
    vector<string> names = {
        "Ace","Two","Three","Four","Five","Six","Seven",
        "Eight","Nine","Ten","Jack","Queen","King"
    };

    vector<TrainImage> ranks;

    for (const auto& name : names) {
        TrainImage t;
        t.name = name;
        t.img = imread(path + "/" + name + ".jpg", IMREAD_GRAYSCALE);

        if (t.img.empty())
            cout << "Missing rank: " << name << endl;

        ranks.push_back(t);
    }

    return ranks;
}

vector<TrainImage> loadSuits(const string& path) {
    vector<string> names = { "Spades","Diamonds","Clubs","Hearts" };

    vector<TrainImage> suits;

    for (const auto& name : names) {
        TrainImage t;
        t.name = name;
        t.img = imread(path + "/" + name + ".jpg", IMREAD_GRAYSCALE);

        if (t.img.empty())
            cout << "Missing suit: " << name << endl;

        suits.push_back(t);
    }

    return suits;
}

/////////////////////////////
// Preprocess Image
/////////////////////////////

Mat preprocessImage(const Mat& image) {
    Mat gray, blurImg, thresh;

    cvtColor(image, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, blurImg, Size(5, 5), 0);

    int bkg_level = gray.at<uchar>(gray.rows / 100, gray.cols / 2);
    int thresh_level = bkg_level + BKG_THRESH;

    threshold(blurImg, thresh, thresh_level, 255, THRESH_BINARY);

    return thresh;
}

/////////////////////////////
// Find Cards
/////////////////////////////

vector<vector<Point>> findCards(const Mat& thresh) {
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    findContours(thresh, contours, hierarchy,
        RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    vector<vector<Point>> cards;

    for (const auto& contour : contours) {

        double area = contourArea(contour);

        if (area < CARD_MIN_AREA || area > CARD_MAX_AREA)
            continue;

        vector<Point> approx;
        approxPolyDP(contour,
            approx,
            arcLength(contour, true) * 0.02,
            true);

        if (approx.size() == 4 && isContourConvex(approx)) {
            cards.push_back(approx);
        }
    }

    return cards;
}

/////////////////////////////
// Flatten Card
/////////////////////////////

Mat flattener(const Mat& image,
    const vector<Point2f>& pts,
    int /*w*/,
    int /*h*/)
{
    vector<Point2f> rect = orderPoints(pts);

    vector<Point2f> dst = {
        Point2f(0,0),
        Point2f(199,0),
        Point2f(199,299),
        Point2f(0,299)
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

QueryCard preprocessCard(const vector<Point>& contour, const Mat& image)
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
    threshold(qCard.warp, warpThresh, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);

    Mat corner = warpThresh(Rect(0, 0, CORNER_WIDTH, CORNER_HEIGHT)).clone();

    int split = static_cast<int>(corner.rows * 0.6f);

    Mat rank = corner(Rect(0, 0, corner.cols, split)).clone();
    Mat suit = corner(Rect(0, split, corner.cols, corner.rows - split)).clone();

    // Rank extraction
    {
        Mat rankCopy = rank.clone();
        vector<vector<Point>> rank_cnts;
        findContours(rankCopy, rank_cnts, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        if (!rank_cnts.empty()) {
            sort(rank_cnts.begin(), rank_cnts.end(),
                [](const auto& a, const auto& b) {
                    return contourArea(a) > contourArea(b);
                });

            Rect r = boundingRect(rank_cnts[0]);
            r &= Rect(0, 0, rank.cols, rank.rows);

            if (r.area() > 0) {
                Mat roi = rank(r);
                resize(roi, qCard.rank_img, Size(RANK_WIDTH, RANK_HEIGHT));
            }
        }
    }

    // Suit extraction
    {
        Mat suitCopy = suit.clone();
        vector<vector<Point>> suit_cnts;
        findContours(suitCopy, suit_cnts, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        if (!suit_cnts.empty()) {
            sort(suit_cnts.begin(), suit_cnts.end(),
                [](const auto& a, const auto& b) {
                    return contourArea(a) > contourArea(b);
                });

            Rect r = boundingRect(suit_cnts[0]);
            r &= Rect(0, 0, suit.cols, suit.rows);

            if (r.area() > 0) {
                Mat roi = suit(r);
                resize(roi, qCard.suit_img, Size(SUIT_WIDTH, SUIT_HEIGHT));
            }
        }
    }

    return qCard;
}

/////////////////////////////
// Match Card
/////////////////////////////

void matchCard(QueryCard& qCard,
    const vector<TrainImage>& ranks,
    const vector<TrainImage>& suits)
{
    int best_rank_diff = 100000;
    int best_suit_diff = 100000;

    for (const auto& r : ranks) {
        if (qCard.rank_img.empty() || r.img.empty()) continue;

        Mat trainThresh;
        threshold(r.img, trainThresh, 0, 255, THRESH_BINARY | THRESH_OTSU);

        Mat trainResized;
        resize(trainThresh, trainResized, Size(RANK_WIDTH, RANK_HEIGHT));

        Mat diff;
        absdiff(qCard.rank_img, trainResized, diff);

        int d = static_cast<int>(sum(diff)[0] / 255.0);

        if (d < best_rank_diff) {
            best_rank_diff = d;
            qCard.best_rank_match = r.name;
        }
    }

    for (const auto& s : suits) {
        if (qCard.suit_img.empty() || s.img.empty()) continue;

        Mat trainThresh;
        threshold(s.img, trainThresh, 0, 255, THRESH_BINARY | THRESH_OTSU);

        Mat trainResized;
        resize(trainThresh, trainResized, Size(SUIT_WIDTH, SUIT_HEIGHT));

        Mat diff;
        absdiff(qCard.suit_img, trainResized, diff);

        int d = static_cast<int>(sum(diff)[0] / 255.0);

        if (d < best_suit_diff) {
            best_suit_diff = d;
            qCard.best_suit_match = s.name;
        }
    }

    if (best_rank_diff > RANK_DIFF_MAX) qCard.best_rank_match = "Unknown";
    if (best_suit_diff > SUIT_DIFF_MAX) qCard.best_suit_match = "Unknown";

    qCard.rank_diff = best_rank_diff;
    qCard.suit_diff = best_suit_diff;
}

/////////////////////////////
// Draw Results
/////////////////////////////

void drawResults(Mat& image, const QueryCard& qCard) {
    circle(image, qCard.center, 5, Scalar(255, 0, 0), FILLED);

    string text = qCard.best_rank_match + " of " + qCard.best_suit_match;

    putText(image,
        text,
        Point(qCard.center.x - 60, qCard.center.y),
        FONT_HERSHEY_SIMPLEX,
        0.8,
        Scalar(0, 255, 0),
        2);
}

/////////////////////////////
// Main
/////////////////////////////

int main() {
    string imagePath = "testimage2.webp";
    Mat image = imread(imagePath);

    if (image.empty()) {
        cout << "Could not open image." << endl;
        return -1;
    }

    vector<TrainImage> trainRanks = loadRanks("Card_Imgs");
    vector<TrainImage> trainSuits = loadSuits("Card_Imgs");

    Mat thresh = preprocessImage(image);

    vector<vector<Point>> cards = findCards(thresh);

    cout << "Cards found: " << cards.size() << endl;

    for (const auto& c : cards) {
        QueryCard qCard = preprocessCard(c, image);
        matchCard(qCard, trainRanks, trainSuits);
        drawResults(image, qCard);

        cout << qCard.best_rank_match
            << " of "
            << qCard.best_suit_match << endl;
    }

    imshow("Detected Cards", image);
    waitKey(0);

    return 0;
}
