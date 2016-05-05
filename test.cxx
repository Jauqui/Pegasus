/************************************************************************/
/*                                                                      */
/*    Copyright 2013-2014 by Ullrich Koethe and Stuart Berg             */
/*                                                                      */
/*    This file is part of the VIGRA computer vision library.           */
/*    The VIGRA Website is                                              */
/*        http://hci.iwr.uni-heidelberg.de/vigra/                       */
/*    Please direct questions, bug reports, and contributions to        */
/*        ullrich.koethe@iwr.uni-heidelberg.de    or                    */
/*        vigra@informatik.uni-hamburg.de                               */
/*                                                                      */
/*    Permission is hereby granted, free of charge, to any person       */
/*    obtaining a copy of this software and associated documentation    */
/*    files (the "Software"), to deal in the Software without           */
/*    restriction, including without limitation the rights to use,      */
/*    copy, modify, merge, publish, distribute, sublicense, and/or      */
/*    sell copies of the Software, and to permit persons to whom the    */
/*    Software is furnished to do so, subject to the following          */
/*    conditions:                                                       */
/*                                                                      */
/*    The above copyright notice and this permission notice shall be    */
/*    included in all copies or substantial portions of the             */
/*    Software.                                                         */
/*                                                                      */
/*    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND    */
/*    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES   */
/*    OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND          */
/*    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT       */
/*    HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,      */
/*    WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING      */
/*    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR     */
/*    OTHER DEALINGS IN THE SOFTWARE.                                   */
/*                                                                      */
/************************************************************************/

#include <vigra/unittest.hxx>
#include <vigra/matrix.hxx>
#include <vigra/pegasos.hxx>

#include "testdata.hxx"

using namespace vigra;

struct PegasosTest
{
    void test_pegasos()
    {
        // Initialize pegasos
        Matrix<double> features(Shape2(150, 5), iris_dataset_features);
        Matrix<int>    labels(Shape2(150, 1), iris_dataset_labels);
        int            T(2000);
        int            k(150);
        
        double         lambda(0.01);

        double          epsilon(8);

        double error = 0.0;
        Pegasos pegasos(features, labels, lambda, T, k);

        Matrix<int> predictions(Shape2(labels.size(0), labels.size(1)));
        predictions.init(1);
        pegasos.predictLabels(features, predictions);

        std::cout << "Predictions: " << std::endl;
        for (int i=0; i<predictions.size(0); i++)
            std::cout << predictions[i] << ", ";
        std::cout << std::endl;

        for (int i=0; i<predictions.size(0); i++)
            if (predictions[i] != labels[i]) error += 1;

        error *= 100.0 / labels.size(0);
        std::cout << "Error: " << error << "%" << std::endl;

        std::ostringstream s;
        s << "failure in Pegasos SVM. Error to big (" << error << "%)";
        shouldMsg(error < epsilon, s.str().c_str());
    }


    void test_sizes()
    {
        Matrix<double> features(Shape2(149, 5), iris_dataset_features);
        Matrix<int>    labels(Shape2(150, 1), iris_dataset_labels);

        int            T(500);
        int            k(50);

        double         lambda(0.01);

        try
        {
            Pegasos pegasos(features, labels, lambda, T, k);
        }
        catch (std::exception & e)
        {
            std::string error = e.what();

            error = error.substr(0, 27);
            if (error.compare("Number of samples missmatch"))
                shouldMsg(0, e.what());
        }
    }


    void test_subset_size()
    {
        Matrix<double> features(Shape2(150, 5), iris_dataset_features);
        Matrix<int>    labels(Shape2(150, 1), iris_dataset_labels);

        int            T(500);
        int            k(151);

        double         lambda(0.01);

        try
        {
            Pegasos pegasos(features, labels, lambda, T, k);
        }
        catch (std::exception & e)
        {
            std::string error = e.what();

            error = error.substr(0, 24);
            if (error.compare("The subset size selected"))
                shouldMsg(0, e.what());
        }
    }

    void test_pegasos_predictions_missmatch()
    {
        // Initialize pegasos
        Matrix<double> features(Shape2(150, 5), iris_dataset_features);
        Matrix<int>    labels(Shape2(150, 1), iris_dataset_labels);
        int            T(500);
        int            k(50);

        double         lambda(0.01);

        double          epsilon(0.08);

        double error = 0.0;
        Pegasos pegasos(features, labels, lambda, T, k);

        features = Matrix<double>(Shape2(150, 4), iris_dataset_features);
        Matrix<int> predictions(Shape2(labels.size(0), labels.size(1)));
        predictions.init(1);
        try
        {
            pegasos.predictLabels(features, predictions);
        }
        catch (std::exception & e)
        {
            std::string error = e.what();

            error = error.substr(0, 28);
            if (error.compare("Number of features missmatch"))
                shouldMsg(0, e.what());
        }
    }
};


struct PegasosTestSuite
: public test_suite
{
    PegasosTestSuite()
    : test_suite("PegasosTestSuite")
    {
        add( testCase( &PegasosTest::test_sizes));
        add( testCase( &PegasosTest::test_subset_size));
        add( testCase( &PegasosTest::test_pegasos_predictions_missmatch));

        add( testCase( &PegasosTest::test_pegasos));
    }
};


int main(int argc, char ** argv)
{
    PegasosTestSuite test;

    int failed = test.run(testsToBeExecuted(argc, argv));

    std::cout << test.report() << std::endl;
    return (failed != 0);
}
