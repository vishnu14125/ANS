#include <iostream>
#include <string>
using namespace std;

#define d 256 // Number of characters in the input alphabet
#define q 101 // A prime number for hashing

void rabinKarp(string text, string pattern) {
    int n = text.length();
    int m = pattern.length();
    int h = 1; // Hash value for the pattern
    int p = 0, t = 0; // Hash values for the pattern and text

    // Compute h = pow(d, m-1) % q
    for (int i = 0; i < m - 1; i++)
        h = (h * d) % q;

    // Calculate initial hash values
    for (int i = 0; i < m; i++) {
        p = (d * p + pattern[i]) % q;
        t = (d * t + text[i]) % q;
    }

    // Slide the pattern over the text
    for (int i = 0; i <= n - m; i++) {
        // Check the hash values
        if (p == t) {
            // Check for exact match
            bool match = true;
            for (int j = 0; j < m; j++) {
                if (text[i + j] != pattern[j]) {
                    match = false;
                    break;
                }
            }
            if (match)
                cout << "Pattern found at index " << i << endl;
        }

        // Calculate the hash for the next window
        if (i < n - m) {
            t = (d * (t - text[i] * h) + text[i + m]) % q;
            if (t < 0)
                t += q; // Adjust if the hash becomes negative
        }
    }
}

int main() {
    string text, pattern;
    cout << "Enter the text: ";
    getline(cin, text);
    cout << "Enter the pattern: ";
    getline(cin, pattern);

    rabinKarp(text, pattern);
    return 0;
}


#include <iostream>
#include <vector>
#include <string>
using namespace std;

// Compute the longest prefix suffix (LPS) array
void computeLPSArray(string pattern, vector<int>& lps) {
    int m = pattern.length(), len = 0;
    lps[0] = 0; // lps[0] is always 0

    int i = 1;
    while (i < m) {
        if (pattern[i] == pattern[len]) {
            len++;
            lps[i] = len;
            i++;
        } else {
            if (len != 0) {
                len = lps[len - 1];
            } else {
                lps[i] = 0;
                i++;
            }
        }
    }
}

// KMP Pattern Searching algorithm
void KMP(string text, string pattern) {
    int n = text.length(), m = pattern.length();
    vector<int> lps(m);

    // Preprocess the pattern
    computeLPSArray(pattern, lps);

    int i = 0, j = 0; // i for text, j for pattern
    while (i < n) {
        if (pattern[j] == text[i]) {
            i++;
            j++;
        }

        if (j == m) {
            cout << "Pattern found at index " << i - j << endl;
            j = lps[j - 1];
        } else if (i < n && pattern[j] != text[i]) {
            if (j != 0) {
                j = lps[j - 1];
            } else {
                i++;
            }
        }
    }
}

int main() {
    string text, pattern;
    cout << "Enter the text: ";
    getline(cin, text);
    cout << "Enter the pattern: ";
    getline(cin, pattern);

    KMP(text, pattern);
    return 0;
}
#include <iostream>
#include <string>
#include <vector>
using namespace std;

// Function to preprocess the string and add boundaries
string preprocessString(const string& s) {
    string t = "@";
    for (char c : s) {
        t += "#" + string(1, c);
    }
    t += "#$";
    return t;
}

// Manacher's algorithm
string manacher(const string& s) {
    string t = preprocessString(s);
    int n = t.size();
    vector<int> p(n, 0); // Palindrome radii
    int center = 0, right = 0;

    for (int i = 1; i < n - 1; i++) {
        // Mirror of i with respect to center
        int mirror = 2 * center - i;

        if (i < right)
            p[i] = min(right - i, p[mirror]);

        // Expand around center i
        while (t[i + 1 + p[i]] == t[i - 1 - p[i]])
            p[i]++;

        // Update center and right if the palindrome expands past right
        if (i + p[i] > right) {
            center = i;
            right = i + p[i];
        }
    }

    // Find the longest palindrome
    int maxLength = 0, centerIndex = 0;
    for (int i = 1; i < n - 1; i++) {
        if (p[i] > maxLength) {
            maxLength = p[i];
            centerIndex = i;
        }
    }

    // Extract the longest palindrome from the original string
    int start = (centerIndex - maxLength) / 2;
    return s.substr(start, maxLength);
}

int main() {
    string s;
    cout << "Enter the string: ";
    getline(cin, s);

    string result = manacher(s);
    cout << "Longest palindromic substring: " << result << endl;

    return 0;
}
#include <iostream>
#include <vector>
using namespace std;

// Function to heapify a subtree rooted at index i
void heapify(vector<int>& arr, int n, int i) {
    int largest = i;          // Initialize largest as root
    int left = 2 * i + 1;     // Left child index
    int right = 2 * i + 2;    // Right child index

    // If left child is larger than root
    if (left < n && arr[left] > arr[largest])
        largest = left;

    // If right child is larger than largest so far
    if (right < n && arr[right] > arr[largest])
        largest = right;

    // If largest is not root
    if (largest != i) {
        swap(arr[i], arr[largest]); // Swap
        heapify(arr, n, largest);  // Recursively heapify the affected subtree
    }
}

// Heap Sort function
void heapSort(vector<int>& arr) {
    int n = arr.size();

    // Build heap (rearrange array)
    for (int i = n / 2 - 1; i >= 0; i--)
        heapify(arr, n, i);

    // Extract elements from heap one by one
    for (int i = n - 1; i > 0; i--) {
        swap(arr[0], arr[i]); // Move current root to end
        heapify(arr, i, 0);   // Call heapify on the reduced heap
    }
}

int main() {
    int n;
    cout << "Enter the number of elements: ";
    cin >> n;

    vector<int> arr(n);
    cout << "Enter the elements: ";
    for (int i = 0; i < n; i++)
        cin >> arr[i];

    heapSort(arr);

    cout << "Sorted array: ";
    for (int x : arr)
        cout << x << " ";
    cout << endl;

    return 0;
}
#include <iostream>
#include <vector>
using namespace std;

// Function to partition the array
int partition(vector<int>& arr, int low, int high) {
    int pivot = arr[high]; // Choose the last element as the pivot
    int i = low - 1;       // Index of smaller element

    for (int j = low; j < high; j++) {
        if (arr[j] < pivot) {
            i++;
            swap(arr[i], arr[j]); // Swap if element is smaller than pivot
        }
    }
    swap(arr[i + 1], arr[high]); // Place pivot in its correct position
    return i + 1;
}

// QuickSort function
void quickSort(vector<int>& arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high); // Partitioning index

        // Recursively sort elements before and after partition
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

int main() {
    int n;
    cout << "Enter the number of elements: ";
    cin >> n;

    vector<int> arr(n);
    cout << "Enter the elements: ";
    for (int i = 0; i < n; i++)
        cin >> arr[i];

    quickSort(arr, 0, n - 1);

    cout << "Sorted array: ";
    for (int x : arr)
        cout << x << " ";
    cout << endl;

    return 0;
}
#include <iostream>
#include <vector>
#include <climits>
using namespace std;

const int INF = INT_MAX;

// Function to solve TSP using DP and Bitmasking
int tsp(int mask, int pos, const vector<vector<int>>& dist, vector<vector<int>>& dp) {
    int n = dist.size();

    if (mask == (1 << n) - 1) // All cities visited
        return dist[pos][0];  // Return to the starting city

    if (dp[mask][pos] != -1) // Already computed
        return dp[mask][pos];

    int ans = INF;
    for (int city = 0; city < n; city++) {
        if ((mask & (1 << city)) == 0) { // If the city is not visited
            int newAns = dist[pos][city] + tsp(mask | (1 << city), city, dist, dp);
            ans = min(ans, newAns);
        }
    }
    return dp[mask][pos] = ans;
}

int main() {
    int n;
    cout << "Enter the number of cities: ";
    cin >> n;

    vector<vector<int>> dist(n, vector<int>(n));
    cout << "Enter the distance matrix:\n";
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            cin >> dist[i][j];

    vector<vector<int>> dp(1 << n, vector<int>(n, -1)); // DP table

    cout << "Minimum cost of visiting all cities: " << tsp(1, 0, dist, dp) << endl;

    return 0;
}
#include <iostream>
#include <vector>
#include <algorithm>
#include <climits>
using namespace std;

// Function to calculate the total cost of a given assignment
int calculateCost(const vector<vector<int>>& costMatrix, const vector<int>& assignment) {
    int totalCost = 0;
    int n = costMatrix.size();
    for (int i = 0; i < n; i++) {
        totalCost += costMatrix[i][assignment[i]];
    }
    return totalCost;
}

// Brute-force solution to the Assignment Problem
int assignmentProblem(const vector<vector<int>>& costMatrix) {
    int n = costMatrix.size();
    vector<int> assignment(n);
    for (int i = 0; i < n; i++) {
        assignment[i] = i; // Initialize assignment [0, 1, 2, ...]
    }

    int minCost = INT_MAX;
    do {
        // Calculate the cost for the current assignment
        int currentCost = calculateCost(costMatrix, assignment);
        minCost = min(minCost, currentCost);
    } while (next_permutation(assignment.begin(), assignment.end()));

    return minCost;
}

int main() {
    int n;
    cout << "Enter the number of workers/tasks: ";
    cin >> n;

    vector<vector<int>> costMatrix(n, vector<int>(n));
    cout << "Enter the cost matrix:\n";
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cin >> costMatrix[i][j];
        }
    }

    cout << "Minimum cost of assignment: " << assignmentProblem(costMatrix) << endl;

    return 0;
}
#include <iostream>
#include <vector>
#include <climits>
using namespace std;

// Function to run the Floyd-Warshall algorithm
void floydWarshall(vector<vector<int>>& dist) {
    int n = dist.size();

    // Main Floyd-Warshall algorithm
    for (int k = 0; k < n; k++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (dist[i][k] != INT_MAX && dist[k][j] != INT_MAX) {
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]);
                }
            }
        }
    }
}

int main() {
    int n;
    cout << "Enter the number of vertices: ";
    cin >> n;

    // Initialize the distance matrix
    vector<vector<int>> dist(n, vector<int>(n, INT_MAX));

    cout << "Enter the adjacency matrix (use 0 for no edge and -1 for diagonal elements):\n";
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cin >> dist[i][j];
            if (dist[i][j] == -1 && i != j)
                dist[i][j] = INT_MAX; // No edge
            if (i == j) dist[i][j] = 0; // Distance to self is 0
        }
    }

    // Run Floyd-Warshall algorithm
    floydWarshall(dist);

    // Output the shortest distance matrix
    cout << "Shortest distance matrix:\n";
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (dist[i][j] == INT_MAX)
                cout << "INF ";
            else
                cout << dist[i][j] << " ";
        }
        cout << endl;
    }

    return 0;
}
#include <iostream>
#include <vector>
#include <climits>
using namespace std;

// Function to find the minimum number of coins required to make the given amount
int coinChange(vector<int>& coins, int amount) {
    // dp[i] will store the minimum number of coins required to make the amount i
    vector<int> dp(amount + 1, INT_MAX);
    dp[0] = 0; // Base case: 0 coins needed to make amount 0

    // Loop through all amounts from 1 to the given amount
    for (int i = 1; i <= amount; i++) {
        // Try every coin denomination
        for (int coin : coins) {
            if (i - coin >= 0 && dp[i - coin] != INT_MAX) {
                dp[i] = min(dp[i], dp[i - coin] + 1);
            }
        }
    }

    // If dp[amount] is still INT_MAX, return -1 (indicating no solution)
    return dp[amount] == INT_MAX ? -1 : dp[amount];
}

int main() {
    int n, amount;
    cout << "Enter the number of coin denominations: ";
    cin >> n;

    vector<int> coins(n);
    cout << "Enter the coin denominations: ";
    for (int i = 0; i < n; i++) {
        cin >> coins[i];
    }

    cout << "Enter the amount: ";
    cin >> amount;
    int result = coinChange(coins, amount);
    if (result == -1)
        cout << "It's not possible to make the given amount with the given coins." << endl;
    else
        cout << "Minimum number of coins required: " << result << endl;

    return 0;
}
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

// Struct to represent an activity
struct Activity {
    int start;
    int end;
};

// Function to perform the activity selection
void activitySelection(vector<Activity>& activities) {
    // Sort the activities based on their end time
    sort(activities.begin(), activities.end(), [](Activity a, Activity b) {
        return a.end < b.end;
    });

    // The first activity is always selected
    int lastSelected = 0;
    cout << "Selected activities are: \n";
    cout << "(" << activities[lastSelected].start << ", " << activities[lastSelected].end << ")\n";

    // Consider the rest of the activities
    for (int i = 1; i < activities.size(); i++) {
        // If the start time of the current activity is greater than or equal to the finish time of the last selected activity
        if (activities[i].start >= activities[lastSelected].end) {
            cout << "(" << activities[i].start << ", " << activities[i].end << ")\n";
            lastSelected = i; // Update the last selected activity
        }
    }
}

int main() {
    int n;
    cout << "Enter the number of activities: ";
    cin >> n;

    vector<Activity> activities(n);
    cout << "Enter the start and end times of activities:\n";
    for (int i = 0; i < n; i++) {
        cout << "Activity " << i + 1 << " (start time, end time): ";
        cin >> activities[i].start >> activities[i].end;
    }

    activitySelection(activities);

    return 0;
}
#include <iostream>
#include <vector>
using namespace std;

void sieveOfSundaram(int n) {
    // Calculate the limit for the sieve
    int nNew = (n - 1) / 2;

    // Create a boolean array to mark numbers as non-prime
    vector<bool> marked(nNew + 1, false);

    // Perform the sieve of Sundaram
    for (int i = 1; i <= nNew; i++) {
        for (int j = i; (i + j + 2 * i * j) <= nNew; j++) {
            marked[i + j + 2 * i * j] = true;
        }
    }

    // 2 is a prime number, so we manually add it to the result
    if (n > 2) cout << 2 << " ";

    // The remaining numbers (2i + 1) that are not marked are primes
    for (int i = 1; i <= nNew; i++) {
        if (!marked[i]) {
            cout << 2 * i + 1 << " ";
        }
    }
    cout << endl;
}

int main() {
    int n;
    cout << "Enter the number up to which primes are to be found: ";
    cin >> n;

    cout << "Primes up to " << n << " are: ";
    sieveOfSundaram(n);

    return 0;
}
#include <iostream>
#include <vector>
using namespace std;

// Size of the chessboard
const int N = 8;

// Function to check if a position is inside the chessboard and not yet visited
bool isSafe(int x, int y, const vector<vector<int>>& board) {
    return (x >= 0 && x < N && y >= 0 && y < N && board[x][y] == -1);
}

// Utility function to print the solution board
void printBoard(const vector<vector<int>>& board) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cout << board[i][j] << " ";
        }
        cout << endl;
    }
}

// Function to solve the Knight's Tour problem using backtracking
bool solveKnightTour(int x, int y, int moveCount, vector<vector<int>>& board, const vector<int>& moveX, const vector<int>& moveY) {
    if (moveCount == N * N) {
        return true;  // All squares are visited
    }

    // Try all next moves from the current position
    for (int i = 0; i < 8; i++) {
        int nextX = x + moveX[i];
        int nextY = y + moveY[i];

        if (isSafe(nextX, nextY, board)) {
            board[nextX][nextY] = moveCount;
            if (solveKnightTour(nextX, nextY, moveCount + 1, board, moveX, moveY)) {
                return true;
            }
            // Backtrack if no solution is found
            board[nextX][nextY] = -1;
        }
    }
    return false;
}

int main() {
    vector<vector<int>> board(N, vector<int>(N, -1)); // Create an 8x8 board initialized to -1
    board[0][0] = 0; // Start from the top-left corner

    // Possible moves for a knight on a chessboard
    vector<int> moveX = {2, 1, -1, -2, -2, -1, 1, 2};
    vector<int> moveY = {1, 2, 2, 1, -1, -2, -2, -1};

    if (solveKnightTour(0, 0, 1, board, moveX, moveY)) {
        printBoard(board);
    } else {
        cout << "Solution does not exist!" << endl;
    }

    return 0;
}
#include <iostream>
#include <iostream>
#include <vector>
using namespace std;

// Function to check if there is a subset with sum equal to target
bool subsetSumBacktracking(const vector<int>& arr, int target, int index) {
    // Base cases
    if (target == 0) return true;
    if (index == arr.size() || target < 0) return false;

    // Include the current element and check
    if (subsetSumBacktracking(arr, target - arr[index], index + 1)) {
        return true;
    }

    // Exclude the current element and check
    return subsetSumBacktracking(arr, target, index + 1);
}

int main() {
    vector<int> arr = {3, 34, 4, 12, 5, 2};
    int target = 9;

    if (subsetSumBacktracking(arr, target, 0)) {
        cout << "Subset with sum " << target << " exists." << endl;
    } else {
        cout << "Subset with sum " << target << " does not exist." << endl;
    }

    return 0;
}
#include <iostream>
#include <vector>
using namespace std;

// Function to print the chessboard
void printBoard(const vector<int>& board, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (board[i] == j) {
                cout << "Q "; // Print Queen
            } else {
                cout << ". "; // Empty space
            }
        }
        cout << endl;
    }
    cout << endl;
}

// Function to check if it's safe to place a queen at board[row][col]
bool isSafe(int row, int col, const vector<int>& board) {
    for (int i = 0; i < row; i++) {
        // Check if the queen is in the same column or diagonals
        if (board[i] == col || board[i] - i == col - row || board[i] + i == col + row) {
            return false;
        }
    }
    return true;
}

// Backtracking function to find all solutions to the N-Queens problem
void solveNQueens(int row, vector<int>& board, int N) {
    if (row == N) {
        // All queens are placed successfully
        printBoard(board, N);
        return;
    }

    // Try all columns in the current row
    for (int col = 0; col < N; col++) {
        if (isSafe(row, col, board)) {
            // Place the queen at board[row] = col
            board[row] = col;
            // Recur to place the next queen
            solveNQueens(row + 1, board, N);
        }
    }
}

int main() {
    int N;
    cout << "Enter the value of N (size of the chessboard): ";
    cin >> N;

    vector<int> board(N, -1); // Initialize the board with -1, indicating no queen placed
    solveNQueens(0, board, N); // Start solving from the first row

    return 0;
}


