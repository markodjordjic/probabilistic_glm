import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid
import io
import imageio


class Video:
    """
    Class for making and saving of the video.

    Attributes
    ----------
    collection_of_frames : list
        Container with individual frames. Frame is an image buffered as
        numpy array.
    """
    def __init__(self):
        self.collection_of_frames = []

    def add_frame(self, buffered_image):
        """
        Adds a frame to the collection of frames.

        Parameters
        ----------
        buffered_image : numpy.array
            Image buffered as numpy array.
        """
        self.collection_of_frames.extend([buffered_image])

    def save_video(self, path):
        """
        Save video

        Video is saved to the disk in MP4 format.

        Parameters
        ----------
        path : str
            Absolute path with the file name in which to save video.
            Video is saved in as MP4 file.
        """
        imageio.mimwrite(
            path,
            self.collection_of_frames,
            format='MP4',
            fps=1,
            output_params=['-intra'],
            quality=10
        )


class ProbabilisticGLM:
    """
    Class for initialization, inference of parameters, predication, and
    perstistence of probabilistic General Linear Model (GLM) for
    solving regression problems.

    Attributes
    ----------
    features : numpy.array
        Features to be used for training (samples x features).
    targets : numpy.array
        Targets to be used for training (samples x 1)
    inferred_alpha : float
        Inferred estimate of precision.
    inferred_beta : float
        Inferred estimate of noise.
    inferred_mean : numpy.array
        Inferred estimate of mean coefficients (1 x features).
    inferred_covariance : numpy.array
        Estimate of covariance matrix (features x features).
    sampled_data : numpy.array
        Data sampled from inferred parameters.
    tolerance : float
        Criterion for stopping tessellation, according to the proximity
        between estimated alpha and beta
    training_iterations : int
        Iteration at which training completed.
    mll : float
        Marginal log-likelihood of the model.
    """

    def __init_(self):
        self.features = None
        self.targets = None
        self.inferred_alpha = None
        self.inferred_beta = None
        self.inferred_mean = None
        self.inferred_covariance = None
        self.inferred_inverse_covariance = None
        self.training_iterations = None
        self.tolerance = 1e-6
        self.mll = None

    def get_posterior(self, alpha, beta):
        """
        Compute mean and covariance matrix of the posterior distribution.

        """
        self.inferred_inverse_covariance = (
            alpha
            * np.eye(self.features.shape[1])
            + beta
            * self.features.T.dot(self.features)
        )
        self.inferred_covariance = np.linalg.inv(
            self.inferred_inverse_covariance
        )
        self.inferred_mean = (
            beta
            * (self.inferred_covariance @ np.transpose(self.features))
            @ self.targets
        )

    def tessellate(self,
                   primary_alpha,
                   primary_beta,
                   max_iterations,
                   tolerance,
                   verbose):

        """
        Tessellate the model to the features and targets.

        Parameters
        ----------
        primary_alpha : float
            Initial value of the noise in the data.
        primary_beta : float
            Initial value for the precision of weights.
        max_iterations : int
            Total number of iterations.
        tolerance : float
            Criteria which can be used to end tessellation.
        verbose : bool
            Indication to report or not to report progress.

        Returns
        -------
        alpha : float
            Inferred estimate of precision.
        beta : float
            Inferred estimate of noise.
        mean : numpy.array
           Inferred coefficients of means after tessellation.
        covariance : numpy.array
            Inferred coefficients covariance matrix after tessellation.

        Notes
        -----
        If desired bias needs to be added manually to the features.

        """
        # Get number of samples.
        samples = np.shape(self.features)[0]
        # Compute base eigenvalues.
        primary_eigenvalues = np.linalg.eigvalsh(
            np.transpose(self.features) @ self.features
        )
        # Set values of alpha and beta to initial values.
        self.inferred_alpha = primary_alpha
        self.inferred_beta = primary_beta
        # Tessellate.
        for iteration in range(max_iterations):
            if verbose:
                print('Iteration: %s' % iteration)
            # Set alpha and beta.
            previous_alpha = self.inferred_alpha
            previous_beta = self.inferred_beta
            # Infer mean, covariance, and inverted covariance.
            self.get_posterior(previous_alpha, previous_beta)
            # Compute new eigenvalues.
            eigenvalues = primary_eigenvalues * previous_beta
            # Compute gamma.
            gamma = np.sum(eigenvalues / (eigenvalues + previous_alpha))
            # Compute new alpha.
            self.inferred_alpha = gamma / np.sum(self.inferred_mean**2)
            # Compute new beta.
            beta_inv = 1 / (samples - gamma) * np.sum(
                (self.targets - (self.features@self.inferred_mean))**2)
            self.inferred_beta = 1 / beta_inv
            # Evaluate stopping criterion.
            condition_alpha = np.isclose(
                previous_alpha, self.inferred_alpha, rtol=tolerance
            )
            condition_beta = np.isclose(
                previous_beta, self.inferred_beta, rtol=tolerance
            )
            if condition_alpha and condition_beta:
                break

    def predict(self, features_for_prediction):
        """
        Generates predictions.

        Parameters
        ----------
        features_for_prediction : numpy.array
            Predictors.

        Returns
        -------
        predictions : numpy.array
            Predictive mean.
        uncertainty: numpy.array
            Predictive variance.

        """
        # Compute predictions.
        predictions = features_for_prediction @ self.inferred_mean
        # Compute uncertainty.
        uncertainty = (
            (1/self.inferred_beta)
            + np.sum(
                (features_for_prediction@self.inferred_covariance)
                * features_for_prediction,
                axis=1
            )
        )
        # Return.
        return predictions, uncertainty

    def sample_new_data(self, number_of_samples, features_for_testing):
        """
        Sample (draw) new data

        Data is sampled from inferred mean, covariance, and on the basis
        of features stored in the principal object.

        Parameters
        ----------
        number_of_samples : int
            Number of new samples to be generated.
        features_for_testing : numpy.array
            Features to be used for testing. See notes.

        Returns
        -------
        numpy.array
            Generated data.

        Notes
        -----
        Features for testing should be scaled.
        """
        weights = np.transpose(np.random.multivariate_normal(
            self.inferred_mean.flatten(),
            self.inferred_covariance,
            size=number_of_samples
        ))
        # Place sampled data into the suitable attribute.
        self.sampled_data = features_for_testing @ weights

    def compute_log_marginal_likelihood(self):
        """
        Compute marginal log-likelihood.

        Marginal log-likelihood is suitable for comparison among models
        of different complexities.

        """
        # Get dimensionality of the training set (complexity of the model).
        samples, dimensions = np.shape(self.features)
        # Evidence.
        E_D = (
            self.inferred_beta
            * np.sum(
                (self.targets - self.features.dot(self.inferred_mean))**2
            )
        )
        E_W = self.inferred_alpha * np.sum(self.inferred_mean**2)
        # Place score into appropriate attribute.
        self.mll = .5 * (
            dimensions
            * np.log(self.inferred_alpha)
            + samples
            * np.log(self.inferred_beta)
            - E_D
            - E_W
            - np.log(np.linalg.det(self.inferred_inverse_covariance))
            - samples * np.log(2 * np.pi)
        )

    # def save(self):


def sequential_fit(features_for_training,
                   targets_for_training,
                   features_for_testing,
                   targets_for_testing,
                   steps,
                   video_file,
                   produce_video=False):
    """
    Sequentialy fit the model

    Fitting sequential fitting of the model in order to diagnostify the
    quality of fit. Video fo the fitting process can be saved to disk.

    Parameters
    ----------
    features_for_training : numpy.array
        Features for fitting the model.
    targets_for_training : numpy.array
        Targets for testing the model.
    features_for_testing : numpy.array
        Features for fitting the model.
    targets_for_testing : numpy.array
        Targets for testing the model.
    steps : int
        Size of the step in which model will be fitted.
    produce_video : bool
        Indication if video is to be produced.
    video_file : str
        Absolute path where to the file in which the video will be saved.

    Returns
    -------
    errors : list
        Model performance after every iteration computed as MAE.
    mll : list
        Marginal log-likelihood of the model after every iteration.
    """
    # Declare a video.
    video_of_training = Video()
    # Declare list to receive error statistics.
    errors = []
    # Declare list to receive MLL statistics.
    mll = []
    # Iterate over fitting data set and produce diagnostic plots.
    for sample in range(steps, len(features_for_training), steps):
        # Display message.
        print('Fitting GLM to the training set of size: %s samples.' % sample)
        # Fit model.
        glm = ProbabilisticGLM()
        glm.features = features_for_training[0:sample, :]
        glm.targets = targets_for_training[0:sample]
        glm.tessellate(
            primary_alpha=1.,
            primary_beta=float(1./np.var(targets_for_training)),
            max_iterations=100,
            tolerance=1e-6,
            verbose=False
        )
        glm.compute_log_marginal_likelihood()
        # Generate prediction.
        predictions, uncertainty = glm.predict(
            features_for_prediction=features_for_testing
        )
        # Compute error.
        mae = np.round(np.mean(np.abs(
            targets_for_testing.flatten()
            - predictions.flatten()
        )), decimals=4)
        # Display message.
        print('--- Model achieves error: %s.' % mae)
        errors.append(mae)
        mll.append(glm.mll)
        if produce_video:
            # Draw new samples.
            glm.sample_new_data(
                number_of_samples=5,
                features_for_testing=features_for_testing
            )
            video_of_training.add_frame(produce_plots(
                reference=targets_for_testing,
                prediction=predictions,
                uncertainty=uncertainty,
                generated_data=glm.sampled_data,
                samples=sample,
                error=mae
            ))
    if produce_video:
        # Write video from inventory of images to disk.
        video_of_training.save_video(path=video_file)

    return errors, mll


def produce_plots(reference,
                  prediction,
                  uncertainty,
                  generated_data,
                  samples,
                  error):
    """
    Plot reference, prediction, uncertainty estimate, and sampled data.

    Parameters
    ----------
    reference : numpy.array
        Actual values.
    prediction : numpy.array
        Generated predictions.
    uncertainty : numpy.array
        Point-wise uncertainty estimate.
    generated_data : numpy.array
        Generated data for plotting.
    samples : numpy.array
        Sampled (drawn) data to be plotted.
    error : float
        Error of the model.

    Returns
    -------
    Plot buffered as a numpy array.
    """
    # Set font size.
    plt.rcParams["font.size"] = 6
    # Set figure size.
    figure = plt.figure(figsize=[26.6667*.33, 15*.33])
    # Declare plotting grid.
    plotting_grid = grid.GridSpec(nrows=3, ncols=1, figure=figure)
    # Add plot for prediction and reference.
    axis_1 = figure.add_subplot(plotting_grid[0, 0])
    axis_1.plot(reference, c='r', label='Reference')
    axis_1.plot(prediction, c='b', linestyle=':', label='Prediction')
    axis_1.legend(loc='upper right')
    axis_1.set_title(
        'Tessellation of the model (Training data=%s MAE=%s)' % (samples, error)
    )
    # Add plot for uncertainty.
    axis_2 = figure.add_subplot(plotting_grid[1, 0])
    axis_2.plot(np.sqrt(uncertainty), linestyle=':', label='Uncertainty')
    axis_2.fill_between(
        np.arange(0, len(reference)),
        y1=np.zeros(shape=(len(reference))),
        y2=np.sqrt(uncertainty),
        alpha=.1
    )
    axis_2.legend(loc='upper right')
    axis_2.set_ylim([0, np.max(np.sqrt(uncertainty))*1.5])
    axis_2.set_title('Uncertainty Estimate')
    # Add plot for generated data.
    axis_3 = figure.add_subplot(plotting_grid[2, 0])
    axis_3.plot(generated_data, linewidth=.2)
    axis_3.plot(reference, c='red', label='Reference')
    axis_3.legend(loc='upper right')
    axis_3.set_title('Generated data')
    plt.subplots_adjust(hspace=.33, top=.9, bottom=.1)
    # Declare a buffer.
    buf = io.BytesIO()
    # Place the figure into a buffer.
    plt.savefig(buf, format='raw')
    # Place the buffer into a numpy array.
    image_as_array = np.reshape(
        np.frombuffer(
            buf.getvalue(),
            dtype=np.uint8
        ),
        newshape=(
            int(figure.bbox.bounds[3]),
            int(figure.bbox.bounds[2]), -1
        )
    )
    # Close the figure.
    plt.close('all')
    return image_as_array