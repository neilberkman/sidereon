function p = normcdf(x)
% Standard normal CDF used by the MAAST fixture generator.
p = 0.5 .* erfc(-x ./ sqrt(2));
end
