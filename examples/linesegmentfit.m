% fitting of a line, will be extended to a line segment

do_plot=true;

% uniformly distribute N points on line segment between [lb hb]
lb=-6;
hb=6;
N=50;

% the line is according to Y=alpha+beta*X
% so, we forget about vertical lines for now
alpha=5;
beta=-2;

% we add some noise according to sigma (should be perpendicular to line)
sigma=0.1;

X=unifrnd(lb, hb, 1, N);
Y=alpha+beta*X;
Y=Y+normrnd(0, sigma, 1, N);

P=[X; Y];

if (do_plot)
	figure(1);
	subplot(2,1,1);
	plot(X, Y, '.');
end

function result = loglikelihood(data, alpha, beta, sigma)
	if (sigma < 0)
		result = -Inf;
		return;
	end
	X = data(1,:);
	Y = data(2,:);
	Y_model = alpha + beta * X;
	N = size(X, 2);
	result = -N/2* (log(2*pi*sigma^2)) - sum((Y-Y_model).^2)/(2*sigma^2) ;
end

function result = logprior(alpha, beta, sigma)
	if (sigma < 0)
		result = -Inf;
		return;
	end
	result = -log(sigma) + log(1 + beta^2) * (-3/2);
end

function result = logposterior(data, alpha, beta, sigma)
	result = loglikelihood(data, alpha, beta, sigma) + logprior(alpha, beta, sigma);
end

% model-dimension is 3, we want to get alpha, beta, and sigma (if we do not fix sigma, sampling goes haywire)
dim=3;

% order probability functions in successive order of complexity
logPfuns={@(m)logprior(m(1), m(2), m(3)); @(m)loglikelihood(P, m(1), m(2), m(3)) };

%make a set of starting points for the entire ensemble of walkers
walkers=dim*3;
minit=abs(randn(dim,walkers))

%Apply the MCMC hammer
[models,logP]=gwmcmc(minit,logPfuns,100000);
models(:,:,1:floor(end/5))=[]; %remove 20% as burn-in
models=models(:,:)'; %reshape matrix to collapse the ensemble member dimension

if (do_plot)
	subplot(2,1,2);

	plot3(models(:,1),models(:,2),models(:,3), '.')
	%plot(models(:,1),models(:,2), '.')
end
