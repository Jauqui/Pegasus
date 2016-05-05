/************************************************************************/
/*                                                                      */
/*               Copyright 2013-2014 by Ullrich Koethe                  */
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


#ifndef VIGRA_PEGASOS_HXX
#define VIGRA_PEGASOS_HXX

#include "multi_array.hxx"
#include "random.hxx"

#include <vigra/sampling.hxx>

namespace vigra {

/** \addtogroup MachineLearning Machine Learning
**/
//@{


    /** Linear support vector machine using the pegasos algorithm.
    
        >>> add more documentation here <<<
    */
class Pegasos
{
    public:
        int m_K;
        int m_T;

        double m_L;

        int m_Classes;

        int m_Rows, m_Cols;

        Matrix<double> m_W;


    /**
     * \brief
    */
    template <class U, class C1, class T, class C2>
    void Check( MultiArrayView<2, U, C1>const   & features,
                MultiArrayView<2, T, C2>        & labels,
                int                               subset_size
                )
    {
        std::ostringstream s;

        s << "The subset size selected (" << subset_size << ") is bigger than the number of samples (" <<
             features.size(0) << ")";
        // Check that number of samples is bigger than the subset size
        shouldMsg(subset_size<=features.size(0), s.str().c_str());

        s.str("");
        s << "Number of samples missmatch:" << std::endl << "Labels: " << labels.size(0) << " != features: " <<
             features.size(0) << ")";
        // Check samples match
        shouldMsg(labels.size(0)==features.size(0), s.str().c_str());
    }


    /**
     * \brief
     *
     * \param features:
     *
     * \param labels:
     *
     * \param lambda:
     *
     * \param iterations:
     *
     * \param subset_size:
    */
    template <class U, class C1, class T, class C2>
    Pegasos( MultiArrayView<2, U, C1>const  & features,
             MultiArrayView<2, T, C2>       & labels,
             double                           lambda,
             int                              iterations,
             int                              subset_size
           ) :
        m_K(subset_size),
        m_T(iterations),
        m_L(lambda)

    {
        Check(features, labels, subset_size);

        srand (time(NULL));

        m_Rows = features.size(0);
        m_Cols = features.size(1);

        m_Classes = 0;
        for (int i=0; i<labels.size(0); i++)
            if (labels[i]>m_Classes) m_Classes = labels[i];
        m_Classes++;

        Matrix<int> labelsMatrix = Matrix<int>(Shape2(labels.size(0), m_Classes));
        labelsMatrix.init(0);
        for (int i=0; i<labels.size(0); i++)
            labelsMatrix(i, labels[i]) = 1;

        MersenneTwister randomGenerator;
        learn(features, labelsMatrix, randomGenerator);
    }


    /**
     * \brief Selects a k random components from features and labels
     *
     * \param features:    a N x M matrix containing N samples with M
     *                     features
     *
     * \param labels:      a N x C matrix containing N samples with C
     *                     classes
     *
     * \param subFeatures: subset of k features component's
     *
     * \param subLabels:   subset of k label component's
     *
     **/
    template <class T, class RandomGenerator>
    void SelectSamples( MultiArrayView<2, T>   const & features,
                        MultiArrayView<2, int> const & labels,
                        MultiArrayView<2, T>         & subFeatures,
                        MultiArrayView<2, int>       & subLabels,
                        RandomGenerator const        & random
                )
    {
        Sampler<> sampler( m_Rows,
                    SamplerOptions().withoutReplacement().sampleSize(m_K),
                    &random);
        sampler.sample();

        for(int i = 0; i < sampler.sampledIndices().size(); i++)
        {
            int r = sampler.sampledIndices()[i];

            for (int j=0; j<features.size(1); j++)
                subFeatures(i,j) = features(r, j);

            for (int c=0; c<m_Classes; c++)
                subLabels(i, c) = labels(r, c);
        }
    }


    /**
     * \brief Displays W in the command line (just for debugging)
     *
     */
    void ShowW()
    {
        for (int j=0; j<m_Cols; j++)
        {
            for (int c=0; c<m_Classes; c++)
                std::cout << m_W(c, j) << ", ";
            std::cout << std::endl;
        }
    }

    /**
     * \brief
     *
     * \param features  a N x M matrix containing N samples with M
     *                  features
     *
     * \param labels    a N x C matrix containing N samples with C
     *                  classes
     *
     * \param random    No documentation found!
    */

    template <class T, class RandomGenerator>
    void learn( MultiArrayView<2, T>   const & features,
                MultiArrayView<2, int> const & labels,
                RandomGenerator        const & random)
    {
        // Generate w_1
        m_W = Matrix<double>(Shape2(m_Classes, m_Cols));
        for (int c=0; c<m_Classes; c++)
            for (int j=0; j<m_Cols; j++)
                m_W(c, j) = (double) rand();


        // Normalize to 1 over square root of lambda
        m_W /= (m_W.norm() * sqrt(m_L));

        for (int t=0; t<m_T;t++)
        {
            Matrix<double> subFeatures(Shape2(m_K, m_Cols));
            Matrix<int>    subLabels(Shape2(m_K, m_Classes));
            Matrix<double> subPredictions(Shape2(m_K, m_Classes));

            // Choose a subset of A with size m_K
            SelectSamples(features, labels, subFeatures, subLabels, random);

            // Compute predictions with w at loop t
            double learning_rate = 1 / (m_L * (t+1));

            // Compute predictions for w_t
            subPredictions = subFeatures * m_W.transpose();
            subPredictions /= subPredictions.norm();

            // Set w_t + 0.5
            Matrix<double> multiply = Matrix<double>(Shape2(m_Classes, m_Cols));
            multiply.init(0);

            for (int j=0; j<m_Cols; j++)
                for (int c=0; c<m_Classes; c++)
                    for (int i=0; i<m_K; i++)
                        if (subPredictions(i, c) > 0 && subLabels(i, c) >  0 ||
                            subPredictions(i, c) < 0 && subLabels(i, c) == 0 )
                            multiply(c, j) += subLabels(i, c) * subFeatures(i, j);

            m_W = (1 -learning_rate*m_L)*m_W + learning_rate/m_K*multiply;

            // Scale each class of w to obtain w_t+1
            for (int c=0; c<m_Classes; c++)
            {
                double norm = 0.0;

                for (int j=0; j<m_Cols; j++)
                    norm += pow(m_W(c, j), 2.0);
                norm = sqrt(norm);

                double factor = min(1.0, 1/(sqrt(m_L) * norm));
                for (int j=0; j<m_Cols; j++)
                    m_W(c, j) *= factor;
            }
        }
    }


    /** \brief predict a label given a feature using pegasos algorithm.
     *
     *  \param features: a N x M matrix containing N samples with M
     *                   features
     *
     *  \param labels:   a N x 1 matrxi containing the label of the class
     *                   of each sample
    */
    template <class T>
    void predictLabels(MultiArrayView<2, T>  const & features,
                       MultiArrayView<2, int>      &  labels) const
    {
        std::ostringstream s;
        s << "Number of samples missmatch:" << std::endl << "Labels: " << labels.size(0) << " != features: " <<
             features.size(0) << ")";
        // Check samples match
        shouldMsg(labels.size(0)==features.size(0), s.str().c_str());

        s.str("");
        s << "Number of features missmatch:" << std::endl << "Learned : " << m_Cols << " != new features: " <<
             features.size(1) << ")";
        // Check samples match
        shouldMsg(m_Cols==features.size(1), s.str().c_str());

        labels.init(0);

        std::cout << "Minimum " << std::numeric_limits<double>::min() << std::endl;

        // Compute all the predictions and select the bigger
        for (int i=0; i<features.size(0); i++)
        {
            double max = -std::numeric_limits<double>::min();
            for (int c=0; c<m_Classes; c++)
            {
                double val = 0.0;
                for (int j=0; j<features.size(1); j++)
                    val += features(i, j) * m_W(c, j);

                if (val>max)
                {
                    max = val;
                    labels[i] = c;
                }
            }
        }
        std::cout << std::endl;
    }
};

//@}

} // namespace vigra

#endif // VIGRA_PEGASOS_HXX
