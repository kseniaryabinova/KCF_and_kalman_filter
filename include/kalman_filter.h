#ifndef TEST_KALMAN_FILTER_H
#define TEST_KALMAN_FILTER_H

#include <opencv2/core/mat.hpp>
#include <iostream>

class Kalman{
public:
    Kalman() {
        this->H = (cv::Mat_<float>(2, 4) << 1, 0, 0, 0, 0, 1, 0, 0);

        this->A = (cv::Mat_<float>(4, 4) << 1, 0, 0, 0,
                                            0, 1, 0, 0,
                                            0, 0, 1, 0,
                                            0, 0, 0, 1);

        this->I = (cv::Mat_<float>(4, 4) << 1, 0, 0, 0,
                                            0, 1, 0, 0,
                                            0, 0, 1, 0,
                                            0, 0, 0, 1);

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

    void set_from_genome(float* gene){
        int offset = 0;
        for (size_t i=0; i<this->B.rows; ++i){
            for (size_t j=0; j<this->B.cols; ++j){
                this->B.at<float>(i, j) = gene[offset + i * this->B.rows + j];
//                printf("\n[%zu]=%f\n", offset + i * this->B.rows + j,
//                       this->B.at<float>(i, j));
            }
        }

//        std::cout << this->B << std::endl;

        offset += this->B.rows * this->B.cols;
        for (size_t i=0; i<this->u.rows; ++i){
            for (size_t j=0; j<this->u.cols; ++j){
                this->u.at<float>(i, j) = gene[offset + i * this->u.rows + j];
//                printf("\n[%zu]=%f\n", offset + i * this->u.rows + j,
//                       this->u.at<float>(i, j));
            }
        }
//        std::cout << this->u << std::endl;

        offset += this->u.rows * this->u.cols;
        for (size_t i=0; i<this->S.rows; ++i){
            for (size_t j=0; j<this->S.cols; ++j){
                this->S.at<float>(i, j) = gene[offset + i * this->S.rows + j];
//                printf("\n[%zu]=%f\n", offset + i * this->S.rows + j,
//                       this->S.at<float>(i, j));
            }
        }
//        std::cout << this->S << std::endl;

        offset += this->S.rows * this->S.cols;
        for (size_t i=0; i<this->R.rows; ++i){
            for (size_t j=0; j<this->R.cols; ++j){
                this->R.at<float>(i, j) = gene[offset + i * this->R.rows + j];
//                printf("\n[%zu]=%f\n", offset + i * this->R.rows + j,
//                       this->R.at<float>(i, j));
            }
        }
//        std::cout << this->R << std::endl;

        offset += this->R.rows * this->R.cols;
        for (size_t i=0; i<this->A.rows; ++i){
            for (size_t j=0; j<this->A.cols; ++j){
                this->A.at<float>(i, j) = gene[offset + i * this->A.rows + j];
//                printf("\n[%zu]=%f\n", offset + i * this->A.rows + j,
//                       this->A.at<float>(i, j));
            }
        }
//        std::cout << this->A << std::endl;
//        this->B.data = (unsigned char*)gene;
//        this->u.data = (unsigned char*)(gene + 16);
//        this->S.data = (unsigned char*)(gene + 16 + 4);
//        std::cout << S << std::endl;
//        this->R.data = (unsigned char*)(gene + 16 + 4 + 16);
//        this->A.data = (unsigned char*)(gene + 16 + 4 + 16 + 4);
    }

    void set_A(float T){
        this->A.at<float>(0, 2) = T;
        this->A.at<float>(1, 3) = T;

        this->A.at<float>(0, 0) = 1;
        this->A.at<float>(1, 1) = 1;
        this->A.at<float>(2, 2) = 1;
        this->A.at<float>(3, 3) = 1;
    }

    void set_Q(float T){
        float data[] = {
                float(0.25 * T * T * T * T), 0, float(0.5 * T * T * T), 0,
                0, float(0.25 * T * T * T * T), 0, float(0.5 * T * T * T),
                float(0.5 * T * T * T), 0, float(T * T), 0,
                0, float(0.5 * T * T * T), 0, float(T * T)
        };

        this->Q.release();
        this->Q = cv::Mat(4, 4, cv::DataType<float>::type, data);
    }


    /**
     * только подбирать
     */
    void set_B(){}
    void set_u(){}

    void set_S(){}
    void set_H(){}
    void set_R(){}


    cv::Rect& predict(float T, cv::Rect& box){
        if (is_first){
            prev_box = box;

            X = cv::Mat(4, 1, cv::DataType<float>::type);

            X.at<float>(0) = box.x + float(box.width)/2;
            X.at<float>(1) = box.y + float(box.height)/2;
            X.at<float>(2) = 1;
            X.at<float>(3) = 1;

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
                int(X.at<float>(0) - float(prev_box.width)/2),
                int(X.at<float>(1) - float(prev_box.height)/2),
                prev_box.width,
                prev_box.height);

        return *result;
    }

private:
    void correct(float T, cv::Rect& box){
        cv::Mat H_transpose;
        cv::transpose(this->H, H_transpose);

        cv::Mat inverse_mat;
        cv::invert(this->H * this->S * H_transpose + this->R, inverse_mat);

        this->K = this->S * H_transpose * inverse_mat;

        float data[] = {
                float(box.x) + float(box.width)/2,
                float(box.y) + float(box.height)/2,
        };
        auto Y = cv::Mat(2, 1, cv::DataType<float>::type, data);
        this->X = this->X + this->K * (Y - this->H * this->X);

        this->S = (this->I - this->K * this->H) * this->S;
    }

    void dummy_init_matrices(){
        this->B = (cv::Mat_<float>(4, 4) << 1, 0, 0, 0,
                                            0, 1, 0, 0,
                                            0, 0, 1, 0,
                                            0, 0, 0, 1);

        this->u = (cv::Mat_<float>(4, 1) << 1.2, 1.3, 1.5, 1);

        this->S = (cv::Mat_<float>(4, 4) << 1, 0, 0, 0,
                                            0, 1, 0, 0,
                                            0, 0, 1, 0,
                                            0, 0, 0, 1);

        this->R = (cv::Mat_<float>(2, 2) << 1, 0, 0, 1);
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
