"""CPU-only higher-order transverse-average NEM axis response.

Legendre coefficients use u=2*x/L in [-1,1]. The mean is coefficient zero;
weak tests are P_n(u)/2. This is a numerical prototype, with no positivity
or full three-dimensional point-field guarantee.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import brentq
from .nodal_polynomial_reference import _array, _equilibrate, _positive_scalar


class NEMPolynomialAxisError(ValueError):
    pass


def _roots_on_interval(coeff):
    """Derivative partition/bracketing, retaining even-multiplicity roots.

    Recursively partition by derivative roots; each open subinterval is
    monotone. Values near zero at partition boundaries retain repeated roots.
    Extra near-zero candidates are harmless for an extrema minimum. This is
    floating-point root isolation, not a rigorous interval certificate.
    """
    leg = np.polynomial.legendre
    coeff = leg.legtrim(coeff)
    scale = np.max(abs(coeff))
    if len(coeff) <= 1 or scale == 0:
        return np.empty(0)
    coeff = coeff/scale
    turning = _roots_on_interval(leg.legder(coeff))
    bounds = np.unique(np.r_[-1., turning, 1.])
    tolerance = 128*np.finfo(float).eps*np.sum(abs(coeff))
    evaluate = lambda u: leg.legval(u, coeff)
    roots = [x for x in bounds[1:-1] if abs(evaluate(x)) <= tolerance]
    for left,right in zip(bounds[:-1],bounds[1:]):
        fl,fr = evaluate(left),evaluate(right)
        if fl == 0 and -1 < left < 1:
            roots.append(left)
        if fr == 0 and -1 < right < 1:
            roots.append(right)
        if fl != 0 and fr != 0 and np.signbit(fl) != np.signbit(fr):
            roots.append(brentq(evaluate,left,right,xtol=5e-15,maxiter=200))
    return np.unique(roots)


def profile_minimum(coefficients):
    """Return minimum and its t=x/L location for one Legendre coefficient row."""
    c = _array(coefficients,'coefficients')
    if c.ndim != 1 or not 1 <= len(c) <= 15:
        raise ValueError('coefficients must be a vector of degree at most fourteen')
    scale = np.max(abs(c))
    if scale == 0:
        return 0.,-.5
    normalized = c/scale
    points = np.r_[-1.,_roots_on_interval(np.polynomial.legendre.legder(normalized)),1.]
    values = np.polynomial.legendre.legval(points,normalized)*scale
    if not np.all(np.isfinite(values)):
        raise NEMPolynomialAxisError('nonfinite profile values')
    i = np.argmin(values)
    return float(values[i]),float(points[i]/2)


@dataclass(frozen=True)
class NEMPolynomialAxisResult:
    coefficients: np.ndarray  # G,P+1; Legendre in u=2t
    moments: np.ndarray       # P-2,G
    face_means: np.ndarray    # minus,plus; 2,G
    diagnostics: dict


class NEMPolynomialAxisResponse:
    """Loss/reference response from (qmom-ellmom,Jminus,Jplus) to a_1..a_P.

    C acts on group columns. Currents are positive in the global axis. The
    mean is prescribed separately; mean balance belongs to the caller.
    Inversion is gated by the exact dense scaled 2-norm condition number.
    """
    def __init__(self,D,C,L,degree=4,condition_limit=1e12):
        D,C = _array(D,'D'),_array(C,'C')
        L = _positive_scalar(L,'L')
        limit = _positive_scalar(condition_limit,'condition_limit')
        if isinstance(degree,(bool,np.bool_)) or degree not in (4,6,8,10,12,14):
            raise ValueError('degree must be one of 4,6,8,10,12,14')
        if D.ndim != 1 or not D.size or np.any(D <= 0) or C.shape != (len(D),len(D)):
            raise ValueError('D/C must be positive vector / square matching matrix')
        if limit < 1:
            raise ValueError('condition_limit must be at least one')
        self.degree,self.K,self.G = int(degree),int(degree)-2,len(D)
        self.D,self.C,self.L = D.copy(),C.copy(),L
        P,K,G = self.degree,self.K,self.G
        matrix = np.zeros((P*G,P*G))
        for n in range(1,K+1):
            matrix[(n-1)*G:n*G,(n-1)*G:n*G] = C/(2*(2*n+1))
            for j in range(n+2,P+1,2):
                A = 2*(j*(j+1)-n*(n+1))
                matrix[(n-1)*G:n*G,(j-1)*G:j*G] -= np.diag(D/L**2*A/2)
        for j in range(1,P+1):
            matrix[K*G:(K+1)*G,(j-1)*G:j*G] = np.diag(-D/L*(-1)**(j+1)*j*(j+1))
            matrix[(K+1)*G:,(j-1)*G:j*G] = np.diag(-D/L*j*(j+1))
        if not np.all(np.isfinite(matrix)):
            raise NEMPolynomialAxisError('nonfinite axis matrix')
        scaled,row,column = _equilibrate(matrix)
        try:
            self.condition = float(np.linalg.cond(scaled,2))
            if not np.isfinite(self.condition) or self.condition > limit:
                raise NEMPolynomialAxisError('axis condition exceeds limit')
            response = column[:,None]*np.linalg.solve(scaled,row[:,None]*np.eye(P*G))
        except np.linalg.LinAlgError as exc:
            raise NEMPolynomialAxisError('singular axis response') from exc
        if not np.all(np.isfinite(response)):
            raise NEMPolynomialAxisError('nonfinite axis response')
        self.matrix,self.response = matrix,response
        for x in (self.D,self.C,self.matrix,self.response):
            x.setflags(write=False)

    def solve(self,mean,currents,moment_source=None,leakage_moments=None,rtol=1e-10):
        G,P,K = self.G,self.degree,self.K
        mean = _array(mean,'mean'); currents = _array(currents,'currents')
        q = np.zeros((K,G)) if moment_source is None else _array(moment_source,'moment_source')
        ell = np.zeros((K,G)) if leakage_moments is None else _array(leakage_moments,'leakage_moments')
        rtol = _positive_scalar(rtol,'rtol')
        if mean.shape != (G,) or currents.shape != (2,G) or q.shape != (K,G) or ell.shape != (K,G):
            raise ValueError('axis input shapes must be G,2xG,KxG,KxG')
        rhs = np.r_[(q-ell).ravel(),currents.ravel()]
        a = (self.response@rhs).reshape(P,G)
        coeff = np.column_stack((mean,a.T))
        moments = a[:K]/(2*(2*np.arange(1,K+1)+1))[:,None]
        faces = np.stack((np.polynomial.legendre.legval(-1,coeff.T),np.polynomial.legendre.legval(1,coeff.T)))
        # Independent physical integration, not the response matrix residual.
        u,w = np.polynomial.legendre.leggauss(P+2);w=w/2
        v = np.polynomial.legendre.legvander(u,P).T
        dv = np.stack([np.polynomial.legendre.legval([-1.,1.],np.polynomial.legendre.legder(np.eye(P+1)[j]))*2 for j in range(P+1)])
        ddv = np.stack([np.polynomial.legendre.legval(u,np.polynomial.legendre.legder(np.eye(P+1)[j],2))*4 for j in range(P+1)])
        values = coeff@v;second = coeff@ddv
        tests = v[1:K+1]/2
        weighted = (values*w)@tests.T
        weak = self.C@weighted-(self.D/self.L**2)[:,None]*((second*w)@tests.T)+(ell-q).T
        scale = (abs(self.C)@((abs(values)*w)@abs(tests).T)
                 +(self.D/self.L**2)[:,None]*((abs(second)*w)@abs(tests).T)
                 +abs(ell.T)+abs(q.T)+(self.D*abs(mean)/self.L**2)[:,None]*(abs(tests)@w))
        rel = lambda r,s:float(np.max(abs(r)/np.maximum(s,np.finfo(float).tiny)))
        jp = -self.D[:,None]/self.L*(coeff@dv)
        minima,locations = np.array([profile_minimum(c) for c in coeff]).T
        info = dict(condition_number_2_scaled=self.condition,
                    weak_balance_error=rel(weak,scale),
                    current_balance_error=rel(jp-currents.T,abs(jp)+abs(currents.T)+(self.D*abs(mean)/self.L)[:,None]),
                    mean_balance_error=rel(values@w-mean,np.max(abs(coeff),axis=1)),
                    moment_balance_error=rel(weighted-moments.T,np.max(abs(coeff),axis=1)[:,None]),
                    profile_minima=minima,profile_minimum_locations=locations,
                    negative_profile_count=int(np.sum(minima<0)),minimum_profile_flux=float(min(minima)),
                    polynomial_degree=P,extrema_are_interval_certified=False)
        if any(not np.isfinite(info[k]) or info[k]>rtol for k in ('weak_balance_error','current_balance_error','mean_balance_error','moment_balance_error')):
            raise NEMPolynomialAxisError('independent axis physical checks failed')
        return NEMPolynomialAxisResult(coeff,moments,faces,info)
