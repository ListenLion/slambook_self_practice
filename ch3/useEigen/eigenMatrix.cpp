#include <iostream>
using namespace std;
#include <ctime>

#include <Eigen/Core>
#include <Eigen/Dense>
#define MATRIX_SIZE 50

int main(int argc ,char* argv[])
{
	Eigen::Matrix<float,2,3> matrix_23;
	Eigen::Vector3d v_3d;
	Eigen::Matrix<float,3,1> vd_3d;

	Eigen::Matrix3d matrix_33 = Eigen::Matrix3d::Zero();

	Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> matrix_dynamic;

	Eigen::MatrixXd matrix_x;

	matrix_23 <<1, 2, 3, 4, 5, 6;
        cout<< "matrix_23: \n" <<matrix_23 << "\n"<<endl;

	for(int i=0; i< 2; i++){
           for(int j=0; j< 3; j++)
		cout << matrix_23(i,j) <<"\t";
	    cout << endl;
	}

	cout <<"\n**********************************\n"<< endl;
	
	v_3d<< 4,5,6;
	vd_3d<<3,2,1;
	Eigen::Matrix<double,2,1> result = matrix_23.cast<double>() * v_3d;
	cout << result <<endl;

	Eigen::Matrix<float,2,1> result2 = matrix_23 * vd_3d;

	cout << result2 <<endl;

	cout <<"\n**********************************"<< endl;

	matrix_33 = Eigen::Matrix3d::Random();
	cout <<"\n随机数矩阵:\n" << matrix_33 << endl << endl;

	cout << "\n随机数矩阵 -> 转置:\n" << matrix_33.transpose() << endl;
	cout << "\n随机数矩阵 -> 各元素和:\n" << matrix_33.sum() << endl;
	cout << "\n随机数矩阵 -> 迹:\n" << matrix_33.trace() << endl;
	cout << "\n随机数矩阵 -> 数乘:\n" << 10*matrix_33 << endl;
	cout << "\n随机数矩阵 -> 逆:\n" << matrix_33.inverse() << endl;
	cout << "\n随机数矩阵 -> 行列式:\n"  << matrix_33.determinant() << endl;

	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver(matrix_33.transpose()*matrix_33);
	cout << "\nEigen values = \n" << eigen_solver.eigenvalues() << endl;
	cout << "Eigen vectors = \n" << eigen_solver.eigenvectors() << endl;

	Eigen::Matrix < double,MATRIX_SIZE,MATRIX_SIZE > matrix_NN;
	matrix_NN = Eigen::MatrixXd::Random(MATRIX_SIZE,MATRIX_SIZE);
	Eigen::Matrix<double,MATRIX_SIZE,1> v_nNd;
	v_nNd = Eigen::MatrixXd::Random(MATRIX_SIZE,1);

	clock_t time_stt = clock();
	Eigen::Matrix<double,MATRIX_SIZE,1> x = matrix_NN.inverse()*v_nNd;
	//cout <<"\n" << x << "\n" <<endl;
	cout<< "\ntime use in normal inverse is "<< 1000*(clock()-time_stt)/(double)CLOCKS_PER_SEC << " ms" <<endl;

	time_stt = clock();
	x= matrix_NN.colPivHouseholderQr().solve(v_nNd);
	//cout << "\n" << x << "\n" <<endl;
	cout << "time use in QR decomposition is " << 1000* (clock()- time_stt)/(double)CLOCKS_PER_SEC <<" ms" << endl;
	
	return 0;



}
