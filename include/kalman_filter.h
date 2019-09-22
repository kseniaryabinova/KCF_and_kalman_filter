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
        this->H = cv::Mat(2, 4, cv::DataType<double>::type, H_data);

        double A_data[16] = {
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1
        };
        this->A = cv::Mat(4, 4, cv::DataType<double>::type, A_data);

        double identity_data[16] = {
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1
        };
        this->I = cv::Mat(4, 4, cv::DataType<double>::type, identity_data);

        is_first = true;

        dummy_init_matrices();
    }

    ~Kalman(){
        this->X.release();
        this->A.release();
        this->B.release();
        this->u.release();
        this->S.release();
        this->Q.release();
        this->K.release();
        this->H.release();
        this->R.release();
        this->I.release();
    }

    void set_from_genome(double* gene){
        int offset = 0;
        for (size_t i=0; i<this->B.rows; ++i){
            for (size_t j=0; j<this->B.cols; ++j){
                this->B.at<double>(i, j) = gene[offset + i * this->B.rows + j];
                printf("\n[%zu]=%f\n", offset + i * this->B.rows + j,
                       this->B.at<double>(i, j));
            }
        }
        std::cout << this->B << std::endl;

        offset += this->B.rows * this->B.cols;
        for (size_t i=0; i<this->u.rows; ++i){
            for (size_t j=0; j<this->u.cols; ++j){
                this->u.at<double>(i, j) = gene[offset + i * this->u.rows + j];
                printf("\n[%zu]=%f\n", offset + i * this->u.rows + j,
                       this->u.at<double>(i, j));
            }
        }
        std::cout << this->u << std::endl;

        offset += this->u.rows * this->u.cols;
        for (size_t i=0; i<this->S.rows; ++i){
            for (size_t j=0; j<this->S.cols; ++j){
                this->S.at<double>(i, j) = gene[offset + i * this->S.rows + j];
                printf("\n[%zu]=%f\n", offset + i * this->S.rows + j,
                       this->S.at<double>(i, j));
            }
        }
        std::cout << this->S << std::endl;

        offset += this->S.rows * this->S.cols;
        for (size_t i=0; i<this->R.rows; ++i){
            for (size_t j=0; j<this->R.cols; ++j){
                this->R.at<double>(i, j) = gene[offset + i * this->R.rows + j];
                printf("\n[%zu]=%f\n", offset + i * this->R.rows + j,
                       this->R.at<double>(i, j));
            }
        }
        std::cout << this->R << std::endl;

        offset += this->R.rows * this->R.cols;
        for (size_t i=0; i<this->A.rows; ++i){
            for (size_t j=0; j<this->A.cols; ++j){
                this->A.at<double>(i, j) = gene[offset + i * this->A.rows + j];
                printf("\n[%zu]=%f\n", offset + i * this->A.rows + j,
                       this->R.at<double>(i, j));
            }
        }
        std::cout << this->A << std::endl;
//        this->B.data = (unsigned char*)gene;
//        this->u.data = (unsigned char*)(gene + 16);
//        this->S.data = (unsigned char*)(gene + 16 + 4);
//        std::cout << S << std::endl;
//        this->R.data = (unsigned char*)(gene + 16 + 4 + 16);
//        this->A.data = (unsigned char*)(gene + 16 + 4 + 16 + 4);
    }

    void set_A(double T){
//        double data[] = {
//                1, 0, T, 0,
//                0, 1, 0, T,
//                0, 0, 1, 0,
//                0, 0, 0, 1
//        };
        this->A.at<double>(0, 2) = T;
        this->A.at<double>(1, 3) = T;

        this->A.at<double>(0, 0) = 1;
        this->A.at<double>(1, 1) = 1;
        this->A.at<double>(2, 2) = 1;
        this->A.at<double>(3, 3) = 1;

//        this->A.release();
//        this->A = cv::Mat(4, 4, cv::DataType<double>::type, data);
    }

    void set_Q(double T){
        double data[] = {
                (0.25 * T * T * T * T), 0, (0.5 * T * T * T), 0,
                0, (0.25 * T * T * T * T), 0, (0.5 * T * T * T),
                (0.5 * T * T * T), 0, (T * T), 0,
                0, (0.5 * T * T * T), 0, (T * T)
        };

        this->Q.release();
        this->Q = cv::Mat(4, 4, cv::DataType<double>::type, data);
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

            X = cv::Mat(4, 1, cv::DataType<double>::type);

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
//        std::cout << S << std::endl;
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
        auto Y = cv::Mat(2, 1, cv::DataType<double>::type, data);
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
        B = cv::Mat(4, 4, cv::DataType<double>::type, B_data);

        double u_data[] = {
                1.2, 1.3, 1.5, 1
        };
        u = cv::Mat(4, 1, cv::DataType<double>::type, u_data);

        double S_data[] = {
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1
        };
        S = cv::Mat(4, 4, cv::DataType<double>::type, S_data);

        double R_data[] = {
                1, 0,
                0, 1
        };
        R = cv::Mat(2, 2, cv::DataType<double>::type, R_data);
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
