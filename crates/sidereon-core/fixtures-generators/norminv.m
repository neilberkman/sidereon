function x = norminv(p)
% Standard normal inverse CDF used by the MAAST fixture generator.
x = -sqrt(2) .* erfcinv(2 .* p);
end
