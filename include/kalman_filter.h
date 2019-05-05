#ifndef TEST_KALMAN_FILTER_H
#define TEST_KALMAN_FILTER_H

#include <opencv2/core/mat.hpp>
#include <iostream>

class Kalman{
public:
    Kalman() {
        double H_data[8] = {
                1, 0, 0, 0,
                0, 1, 0, 0
        };
        this->H = cv::Mat(2, 4, CV_64F, H_data);

        double A_data[16] = {
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1
        };
        this->A = cv::Mat(4, 4, CV_64F, A_data);

        double identity_data[16] = {
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1
        };
        this->I = cv::Mat(4, 4, CV_64F, identity_data);

        is_first = true;

//        dummy_init_matrices();
    }

    void set_from_genome(const double* gene){
        std::memcpy(B.data, gene, 16 * sizeof(double));
        printf("1\n");
        std::memcpy(u.data, gene + 16, 4 * sizeof(double));
        printf("2\n");
        std::memcpy(S.data, gene + 16 + 4, 16 * sizeof(double));
        printf("3\n");
        std::memcpy(R.data, gene + 16 + 4 + 16, 4 * sizeof(double));
        printf("4\n");
    }

    void set_A(double T){
        double data[] = {
                1, 0, T, 0,
                0, 1, 0, T,
                0, 0, 1, 0,
                0, 0, 0, 1
        };

        this->A.release();
        this->A = cv::Mat(4, 4, CV_64F, data);
    }

    void set_Q(double T){
        double data[] = {
                (0.25 * T * T * T * T), 0, (0.5 * T * T * T), 0,
                0, (0.25 * T * T * T * T), 0, (0.5 * T * T * T),
                (0.5 * T * T * T), 0, (T * T), 0,
                0, (0.5 * T * T * T), 0, (T * T)
        };

//        double data[] = {
//                1, 0, 0, 0,
//                0, 1, 0, 0,
//                0, 0, 1, 0,
//                0, 0, 0, 1
//        };

        this->Q.release();
        this->Q = cv::Mat(4, 4, CV_64F, data);
    }


    /**
     * только подбирать
     */
    void set_B(){}
    void set_u(){}

    void set_S(){}
    void set_H(){}
    void set_R(){}


    cv::Rect& predict(double T, cv::Rect& box){
        if (is_first){
            prev_box = box;

            X = cv::Mat(4, 1, CV_64F);

            X.at<double>(0) = box.x + double(box.width)/2;
            X.at<double>(1) = box.y + double(box.height)/2;
            X.at<double>(2) = 1;
            X.at<double>(3) = 1;

            is_first = false;
        }

        correct(T, box);

        set_A(T);
        X = A * X + B * u;

        cv::Mat A_transpose;
        cv::transpose(A, A_transpose);
        set_Q(T);
        S = A * S * A_transpose + Q;

        prev_box = box;

        auto* result = new cv::Rect(
                int(X.at<double>(0) - double(prev_box.width)/2),
                int(X.at<double>(1) - double(prev_box.height)/2),
                prev_box.width,
                prev_box.height);

        return *result;
    }

private:
    void correct(double T, cv::Rect& box){
        cv::Mat H_transpose;
        cv::transpose(H, H_transpose);

        cv::Mat inverse_mat;
        cv::invert(H * S * H_transpose + R, inverse_mat);

        K = S * H_transpose * inverse_mat;

        double data[] = {
                double(box.x) + double(box.width)/2,
                double(box.y) + double(box.height)/2,
        };
        auto Y = cv::Mat(2, 1, CV_64F, data);
        X = X + K * (Y - H * X);

        S = (I - K* H) * S;
    }

    void dummy_init_matrices(){
        double B_data[] = {
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1
        };
        B = cv::Mat(4, 4, CV_64F, B_data);

        double u_data[] = {
                1.2, 1.3, 1.5, 1
        };
        u = cv::Mat(4, 1, CV_64F, u_data);

        double S_data[] = {
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1
        };
        S = cv::Mat(4, 4, CV_64F, S_data);

        double R_data[] = {
                1, 0,
                0, 1
        };
        R = cv::Mat(2, 2, CV_64F, R_data);
    }


    cv::Mat X;
    cv::Mat A;
    cv::Mat B;
    cv::Mat u;

    cv::Mat S;
    cv::Mat Q;

    cv::Mat K;
    cv::Mat H;
    cv::Mat R;

    cv::Mat I;

    cv::Rect prev_box;

    bool is_first = true;
};

#endif //TEST_KALMAN_FILTER_H
