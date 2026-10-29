#!/usr/bin/env Rscript
# Prespecified staged Cox analysis. No stage invokes another stage implicitly.
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(yaml))
SCRIPT_FILE <- tryCatch(normalizePath(sys.frame(1)$ofile), error=function(e) NA_character_)
if (!length(SCRIPT_FILE) || is.na(SCRIPT_FILE)) {
  file_arg <- grep('^--file=', commandArgs(FALSE), value=TRUE)
  SCRIPT_FILE <- if(length(file_arg)) normalizePath(sub('^--file=', '', file_arg[1])) else NA_character_
}

sha <- function(path) digest::digest(file=path, algo='sha256', serialize=FALSE)
read_json <- function(path) jsonlite::fromJSON(path, simplifyVector=FALSE)
write_json <- function(path, object) {
  if (file.exists(path)) stop('Exclusive output already exists: ', path)
  dir.create(dirname(path), recursive=TRUE, showWarnings=FALSE)
  jsonlite::write_json(object, path, auto_unbox=TRUE, pretty=TRUE, na='null', digits=16)
}
truth <- function(x) tolower(as.character(x)) %in% c('true','1','t')
num <- function(x) suppressWarnings(as.numeric(x))
need <- function(value, message) if(!isTRUE(value)) stop(message, call.=FALSE)
finite_sd <- function(x, tolerance=1e-12) {
  value <- stats::sd(x)
  need(is.finite(value) && value > tolerance, 'Zero/effectively-zero variation; effect is not estimable')
  value
}

verify_protocol <- function(protocol_path, run_root) {
  p <- yaml::read_yaml(protocol_path)
  need(p$schema_version == 1, 'Unsupported protocol schema')
  need(identical(normalizePath(run_root,mustWork=TRUE),
                 normalizePath(p$paths$run_root,mustWork=TRUE)),
       'Actual run root differs from the canonical protocol run root')
  lock_path <- file.path(run_root, 'protocol', 'PROTOCOL_LOCK.json')
  need(file.exists(lock_path), 'Prospective source/protocol lock is required')
  lock <- read_json(lock_path)
  need(identical(lock$protocol_sha256, sha(protocol_path)), 'Protocol hash differs from source lock')
  need(identical(lock$code_sha256[['analyze.R']], sha(SCRIPT_FILE)), 'Statistical code hash differs from source lock')
  need(identical(p$models$tie_handling, 'efron'), 'Only the prespecified Efron tie method is supported')
  need(p$models$primary_full_slopes == 6 && p$models$temporal_reduced_slopes == 7 &&
       p$models$temporal_full_slopes == 8 && p$models$fallback_slopes == 4, 'Prespecified parameter counts changed')
  need(p$simulation$simulations_per_scenario == 1000 && p$bootstrap$association_patient_resamples == 1000,
       'Bounded simulation/bootstrap inventory differs from protocol')
  need(p$simulation$fallback_simulations_per_scenario == 1000 && p$bootstrap$fallback_association_patient_resamples == 1000,
       'Fallback simulation/bootstrap inventory differs from protocol')
  list(protocol=p, lock=lock, protocol_sha256=sha(protocol_path), lock_path=lock_path)
}

require_qualification <- function(run_root, protocol_sha, fallback=FALSE, paths=NULL) {
  path <- file.path(run_root, 'qualification', if(fallback) 'primary_only_decision.json' else 'paired_decision.json')
  need(file.exists(path), 'Required independent measurement qualification decision is absent')
  result <- read_json(path)
  need(identical(result$protocol_sha256, protocol_sha), 'Qualification belongs to a different protocol')
  need(isTRUE(result$passed), 'Measurement qualification failed: association/design stage forbidden')
  need(!is.null(paths),'Qualified cohort/feature binding paths are required')
  feature_path <- if(fallback) file.path(dirname(paths$phenotype_features_csv),'primary_only_features.csv') else paths$phenotype_features_csv
  need(identical(result$cohort_manifest_sha256,sha(paths$cohort_manifest_csv)) &&
       identical(result$baseline_feature_file_sha256,sha(feature_path)),
       'Cohort or baseline measurements changed since perturbation qualification')
  if(fallback) {
    need(identical(result$feature_name,'phiP'),'Fallback qualification is not for prespecified sphericity')
    need(identical(result$supporting_measurement_file_sha256,sha(paths$phenotype_features_csv)),
         'Fallback supporting intensity/voxel-volume measurements changed since qualification')
  }
  list(path=path, sha256=sha(path), decision=result)
}

read_data <- function(p, which=c('paired','primary_only'), partition=c('development','test')) {
  which <- match.arg(which); partition <- match.arg(partition)
  cohort_path <- p$paths$cohort_manifest_csv
  feature_path <- p$paths$phenotype_features_csv
  c <- read.csv(cohort_path, colClasses='character', check.names=FALSE, na.strings=c('', 'NA', 'NaN'))
  f <- read.csv(feature_path, colClasses='character', check.names=FALSE, na.strings=c('', 'NA', 'NaN'))
  need(!anyDuplicated(c$patient_id) && !anyDuplicated(f$patient_id), 'Nonunique record IDs')
  flag <- if(which == 'paired') 'paired_eligible' else 'primary_only_eligible'
  need(flag %in% names(c), paste('Missing finalized cohort flag:', flag))
  chosen <- truth(c[[flag]]) & c$split %in% if(partition == 'development') c('train','val') else 'test'
  c <- c[chosen,,drop=FALSE]
  need(nrow(c) > 0, 'No eligible records in requested partition')
  need(all(c$patient_id %in% f$patient_id), 'An eligible record lacks a measurement row')
  # Match preserves the cohort order and never silently drops a record.
  f <- f[match(c$patient_id, f$patient_id),,drop=FALSE]
  overlap <- intersect(names(c), names(f)); overlap <- setdiff(overlap,'patient_id')
  f <- f[,setdiff(names(f), overlap),drop=FALSE]
  d <- cbind(c, f[,setdiff(names(f),'patient_id'),drop=FALSE])
  fallback_binding <- NULL
  if(which == 'primary_only') {
    shape_path <- file.path(dirname(feature_path),'primary_only_features.csv')
    need(file.exists(shape_path),'Separately qualified primary-only sphericity table absent')
    shape <- read.csv(shape_path,colClasses='character',check.names=FALSE,na.strings=c('','NA','NaN'))
    need(!anyDuplicated(shape$patient_id) && all(d$patient_id %in% shape$patient_id),'Fallback measurement IDs inconsistent')
    shape <- shape[match(d$patient_id,shape$patient_id),,drop=FALSE]
    need(all(truth(shape$valid)),'An eligible fallback record failed independent shape measurement')
    d$phiP <- num(shape$phiP)
    fallback_binding <- list(path=shape_path,sha256=sha(shape_path))
  }
  numeric_fields <- c('OS_TIME','OS_EVENT','AGE','entry_days','HP','HN','M','D','VP_cm3','VN_cm3',
                      'nP','nN','f0P','f1P','f0N','f1N','original_z_spacing_mm','phiP')
  for(name in intersect(numeric_fields, names(d))) d[[name]] <- num(d[[name]])
  need(all(is.finite(d$OS_TIME) & d$OS_TIME > 0 & d$OS_EVENT %in% c(0,1)), 'Invalid supplied OS')
  need(all(is.finite(d$AGE) & d$SEX %in% c('M','F')), 'Compact clinical eligibility is inconsistent')
  need(all(is.finite(d$VP_cm3) & d$VP_cm3 > 0), 'Invalid primary voxel volume')
  if(which == 'paired') {
    need(all(vapply(d[,c('HP','HN','M','D','VN_cm3'),drop=FALSE], function(x) all(is.finite(x)), logical(1))),
         'Invalid paired measurement')
    need(all(d$VN_cm3 > 0 & d$nP >= 100 & d$nN >= 100), 'Paired support/volume rule violated')
    need(max(abs(d$HP-(d$M-d$D)), abs(d$HN-(d$M+d$D))) <= 1e-6, 'M/D reconstruction failed')
  } else need(all(is.finite(d$phiP) & d$phiP > 0 & d$phiP <= 1 + 1e-6), 'Fallback sphericity is invalid')
  attr(d,'input_bindings') <- list(cohort=list(path=cohort_path,sha256=sha(cohort_path)),
                                  features=list(path=feature_path,sha256=sha(feature_path)),
                                  fallback_shape=fallback_binding)
  time_mode(d) # Validate timing before a design or association stage can start.
  d
}

processing <- function(d, fallback=FALSE) {
  result <- list(age_center=mean(d$AGE), age_scale=10, SEX_reference='F')
  if(fallback) result$SD_phiP <- finite_sd(d$phiP) else {
    result$SD_D <- finite_sd(d$D); result$SD_M <- finite_sd(d$M)
  }
  result
}
transform_data <- function(d, transform) {
  d$AGE10 <- (d$AGE-transform$age_center)/transform$age_scale
  d$male <- as.numeric(d$SEX == 'M')
  d$logVP <- log(d$VP_cm3)
  if('VN_cm3' %in% names(d)) d$logVN <- log(d$VN_cm3)
  d
}
time_mode <- function(d) {
  if(!('entry_days' %in% names(d)) || all(is.na(d$entry_days)))
    return('exploratory_retrospective_diagnosis_OS_timing_unverified')
  need(!any(is.na(d$entry_days)), 'Mixed authentic and unknown entry times require a separately specified analysis')
  need(all(is.finite(d$entry_days) & d$entry_days >= 0 & d$entry_days < d$OS_TIME),
       'Finite nonnegative authentic entry must precede event/censoring')
  'authentic_entry_counting_process'
}
survival_lhs <- function(d, temporal=FALSE) {
  if(temporal) 'Surv(start,stop,event)'
  else if(time_mode(d) == 'authentic_entry_counting_process') 'Surv(entry_days,OS_TIME,OS_EVENT)'
  else 'Surv(OS_TIME,OS_EVENT)'
}

fit_checked <- function(formula, data, expected_slopes=NULL) {
  warnings <- character()
  fit <- tryCatch(withCallingHandlers(coxph(formula, data=data, ties='efron',
      na.action=na.fail, singular.ok=FALSE, x=TRUE, y=TRUE, model=TRUE,
      control=coxph.control(iter.max=50)), warning=function(w) {
        warnings <<- c(warnings, conditionMessage(w)); invokeRestart('muffleWarning')
      }), error=function(e) e)
  if(inherits(fit,'error')) return(list(usable=FALSE,error=conditionMessage(fit),warnings=warnings))
  coeff <- coef(fit); covariance <- vcov(fit)
  okay <- all(is.finite(coeff)) && all(is.finite(covariance)) && all(diag(covariance) > 0) &&
          all(is.finite(fit$loglik)) && !any(grepl('infinite|converg|iteration', warnings, ignore.case=TRUE))
  if(!is.null(expected_slopes)) okay <- okay && length(coeff) == expected_slopes
  if(okay) okay <- min(eigen(covariance, symmetric=TRUE, only.values=TRUE)$values) > 0
  list(usable=okay, fit=fit, coefficients=coeff, covariance=covariance, warnings=warnings,
       error=if(okay) NULL else 'Nonfinite, singular, unconverged or wrong-rank model')
}
fit_pair <- function(d, reduced_terms, added_term, expected_full=NULL, temporal=FALSE) {
  need(!anyDuplicated(d$analysis_row_id), 'Analysis rows must have unique local IDs')
  lhs <- survival_lhs(d, temporal)
  reduced <- fit_checked(as.formula(paste(lhs,'~',paste(reduced_terms,collapse='+'))),d,
                         if(is.null(expected_full)) NULL else expected_full-1)
  full <- fit_checked(as.formula(paste(lhs,'~',paste(c(reduced_terms,added_term),collapse='+'))),d,expected_full)
  if(!reduced$usable || !full$usable) return(list(usable=FALSE,reduced=reduced,full=full,
      error=paste(reduced$error,full$error)))
  need(identical(rownames(reduced$fit$model),rownames(full$fit$model)), 'Nested models used different observations')
  need(length(coef(full$fit))-length(coef(reduced$fit)) == 1, 'Primary comparison is not one degree of freedom')
  statistic <- 2*(full$fit$loglik[2]-reduced$fit$loglik[2])
  need(is.finite(statistic) && statistic >= -1e-7, 'Invalid nested likelihood ordering')
  list(usable=TRUE,reduced=reduced,full=full,LRT=max(0,statistic),df=1,
       p_value=pchisq(max(0,statistic),df=1,lower.tail=FALSE),
       n_patients=length(unique(d$patient_id)),n_rows=nrow(d),deaths=sum(d[[if(temporal) 'event' else 'OS_EVENT']]))
}
primary_terms <- c('AGE10','male','logVP','logVN','M')
primary_pair <- function(d) fit_pair(d,primary_terms,'D',6)

temporal_expand <- function(d, cut=1826) {
  entry <- if(time_mode(d) == 'authentic_entry_counting_process') d$entry_days else rep(0,nrow(d))
  records <- lapply(seq_len(nrow(d)), function(i) {
    t <- d$OS_TIME[i]; e <- entry[i]; event <- d$OS_EVENT[i]
    if(e < cut && t > cut) {
      x <- d[rep(i,2),,drop=FALSE]; x$start <- c(e,cut); x$stop <- c(cut,t)
      x$event <- c(0,event); x$late <- c(0,1)
    } else {
      x <- d[i,,drop=FALSE]; x$start <- e; x$stop <- t; x$event <- event; x$late <- as.numeric(e >= cut)
    }
    x
  })
  out <- do.call(rbind,records); rownames(out) <- NULL
  out$Mlate <- out$M*out$late; out$Dlate <- out$D*out$late
  out$analysis_row_id <- paste0(out$patient_id,'-interval-',seq_len(nrow(out)))
  need(all(out$start < out$stop) && sum(out$event) == sum(d$OS_EVENT), 'Temporal split changed events/risk intervals')
  out
}
temporal_pair <- function(d,cut=1826) {
  x <- temporal_expand(d,cut)
  fit_pair(x,c(primary_terms,'D','Mlate'),'Dlate',8,TRUE)
}

linear_effect <- function(coefficients,covariance,weights,scale=1) {
  names(weights) <- names(weights)
  w <- setNames(rep(0,length(coefficients)),names(coefficients)); w[names(weights)] <- weights
  estimate <- sum(w*coefficients); variance <- as.numeric(t(w)%*%covariance%*%w)
  need(is.finite(variance) && variance > 0, 'Invalid transformed effect covariance')
  se <- sqrt(variance); z <- qnorm(.975)
  list(coefficient=estimate,SE=se,lower95=estimate-z*se,upper95=estimate+z*se,
       reported_scale=scale,log_HR=estimate*scale,HR=exp(estimate*scale),
       HR_lower95=exp((estimate-z*se)*scale),HR_upper95=exp((estimate+z*se)*scale))
}
compartment_effects <- function(pair,sd_D) {
  need(pair$usable, 'Cannot transform an unusable fit')
  b <- coef(pair$full$fit); v <- vcov(pair$full$fit)
  list(primary_per_normalized_intensity_unit=linear_effect(b,v,c(M=.5,D=-.5)),
       nodal_per_normalized_intensity_unit=linear_effect(b,v,c(M=.5,D=.5)),
       difference_per_normalized_intensity_unit=linear_effect(b,v,c(D=1)),
       contrast_HR_per_fixed_development_SD_D=linear_effect(b,v,c(D=1),sd_D))
}
pair_summary <- function(pair,sd_D=NULL) {
  if(!pair$usable) return(list(usable=FALSE,error=pair$error,
      reduced_warnings=pair$reduced$warnings,full_warnings=pair$full$warnings))
  result <- list(usable=TRUE,n_patients=pair$n_patients,n_rows=pair$n_rows,deaths=pair$deaths,
      reduced_slopes=length(coef(pair$reduced$fit)),full_slopes=length(coef(pair$full$fit)),
      LRT=pair$LRT,df=1,p_value=pair$p_value,
      coefficients=as.list(coef(pair$full$fit)),warnings=pair$full$warnings)
  if(!is.null(sd_D)) result$compartment_effects <- compartment_effects(pair,sd_D)
  result
}

calibrate_design <- function(d) {
  mode <- time_mode(d)
  entry <- if(mode == 'authentic_entry_counting_process') d$entry_days else rep(0,nrow(d))
  duration <- d$OS_TIME-entry
  need(all(is.finite(duration) & duration > 0), 'Design calibration requires positive observed at-risk durations')
  fraction <- mean(d$OS_EVENT); median_time <- median(duration)
  need(fraction > 0 && fraction < 1 && median_time > 0, 'Cannot calibrate design event/censor approximation')
  total_rate <- log(2)/median_time; event_rate <- total_rate*fraction
  censor_rate <- total_rate*(1-fraction)
  list(event_rate=event_rate,censor_rate=censor_rate,observed_event_fraction=fraction,
       observed_duration_median=median_time,entry=entry,
       calibration_time_scale=if(mode == 'authentic_entry_counting_process') 'since_authentic_entry' else 'supplied_OS_duration_retrospective_timing_unknown')
}
simulate_times <- function(d,p,transform,scenario) {
  s <- p$simulation; calibration <- calibrate_design(d)
  event_rate <- calibration$event_rate; censor_rate <- calibration$censor_rate; entry <- calibration$entry
  lp <- log(s$nuisance_HR_age_per_10_units)*d$AGE10 + log(s$nuisance_HR_SEX)*d$male +
        log(s$nuisance_HR_log_volume_per_unit)*(d$logVP-mean(d$logVP)+d$logVN-mean(d$logVN)) +
        log(s$nuisance_HR_M_constant)*(d$M-mean(d$M))/transform$SD_M
  beta_D <- if(scenario == 'alternative_D_HR1_5') log(s$alternative_HR_D_per_development_SD) else 0
  lp <- lp + beta_D*(d$D-mean(d$D))/transform$SD_D
  event_time <- rexp(nrow(d),event_rate*exp(lp))
  if(scenario == 'common_M_temporal_stress_D_zero') {
    cut <- p$models$temporal_change_point_days
    late_rate <- event_rate*exp(lp + log(s$temporal_stress_HR_M_late_per_development_SD)*(d$M-mean(d$M))/transform$SD_M)
    early_window <- pmax(0,cut-entry)
    later <- event_time > early_window
    event_time[later] <- early_window[later] + rexp(sum(later),late_rate[later])
  }
  censor_time <- rexp(nrow(d),censor_rate)
  # For verified delayed entry, exponential waiting is generated after entry;
  # its limited approximation is retained in the design receipt.
  out <- d; out$OS_TIME <- entry + pmin(event_time,censor_time)
  out$OS_EVENT <- as.integer(event_time <= censor_time)
  list(data=out,true_standardized_D=beta_D,event_fraction=mean(out$OS_EVENT),
       calibration=calibration[setdiff(names(calibration),'entry')])
}
design_assessment <- function(d,p,transform,simulations=p$simulation$simulations_per_scenario) {
  set.seed(p$simulation$seed)
  all_rows <- list(); summaries <- list()
  for(scenario in unlist(p$simulation$scenarios)) {
    rows <- vector('list',simulations)
    for(i in seq_len(simulations)) {
      generated <- simulate_times(d,p,transform,scenario); x <- generated$data
      ph <- primary_pair(x)
      pair <- if(scenario == 'common_M_temporal_stress_D_zero') temporal_pair(x,p$models$temporal_change_point_days) else ph
      term <- if(scenario == 'common_M_temporal_stress_D_zero') 'Dlate' else 'D'
      row <- data.frame(scenario=scenario,replicate=i,usable=pair$usable,p_value=NA_real_,
          estimate=NA_real_,lower95=NA_real_,upper95=NA_real_,width=NA_real_,
          truth=generated$true_standardized_D,event_fraction=generated$event_fraction,
          PH_six_slope_usable=ph$usable,PH_six_slope_p=if(ph$usable) ph$p_value else NA_real_,
          error=if(pair$usable) '' else pair$error,stringsAsFactors=FALSE)
      if(pair$usable) {
        effect <- linear_effect(coef(pair$full$fit),vcov(pair$full$fit),setNames(1,term),transform$SD_D)
        row$p_value <- pair$p_value; row$estimate <- effect$log_HR
        row$lower95 <- effect$lower95*transform$SD_D; row$upper95 <- effect$upper95*transform$SD_D
        row$width <- row$upper95-row$lower95
      }
      rows[[i]] <- row
    }
    rows <- do.call(rbind,rows); good <- rows[rows$usable,,drop=FALSE]
    usable <- nrow(good)/simulations
    coverage <- if(nrow(good)) mean(good$lower95 <= good$truth & good$upper95 >= good$truth) else NA_real_
    rejection <- if(nrow(good)) mean(good$p_value < p$models$alpha) else NA_real_
    width <- if(nrow(good)) median(good$width) else NA_real_
    checks <- list(usable_fraction=usable >= p$simulation$minimum_usable_fraction,
       CI_coverage=is.finite(coverage) && coverage >= p$simulation$CI_coverage_range[1] && coverage <= p$simulation$CI_coverage_range[2])
    width_is_gating <- scenario %in% unlist(p$simulation$interval_width_gating_scenarios)
    if(width_is_gating) checks$median_interval_width <- is.finite(width) && width <= p$simulation$maximum_median_log_HR_CI_width
    if(scenario != 'alternative_D_HR1_5') checks$null_rejection <- is.finite(rejection) && rejection <= p$simulation$maximum_null_rejection_fraction
    summaries[[scenario]] <- list(simulations=simulations,usable_fraction=usable,rejection_fraction=rejection,
       power_if_alternative=if(scenario == 'alternative_D_HR1_5') rejection else NULL,
       coverage=coverage,median_log_HR_CI_width=width,interval_width_is_gating=width_is_gating,
       checks=checks,passed=all(unlist(checks)),
       six_slope_PH_false_positive_if_temporal_stress=if(scenario == 'common_M_temporal_stress_D_zero')
         mean(rows$PH_six_slope_p[rows$PH_six_slope_usable] < p$models$alpha) else NULL)
    all_rows[[scenario]] <- rows
  }
  list(passed=all(vapply(summaries,function(x) x$passed,logical(1))),scenarios=summaries,
       replicates=do.call(rbind,all_rows))
}

resample_patients <- function(d,index) {
  out <- d[index,,drop=FALSE]
  out$original_patient_id <- out$patient_id
  out$patient_id <- paste0(out$patient_id,'-bootstrap-',seq_len(nrow(out)))
  out$analysis_row_id <- out$patient_id; rownames(out) <- NULL
  out
}
bootstrap_primary <- function(d,p,transform) {
  set.seed(p$bootstrap$association_seed); n <- p$bootstrap$association_patient_resamples
  rows <- vector('list',n)
  for(i in seq_len(n)) {
    x <- resample_patients(d,sample.int(nrow(d),nrow(d),replace=TRUE)); fit <- primary_pair(x)
    row <- data.frame(replicate=i,usable=fit$usable,betaP=NA_real_,betaN=NA_real_,betaD=NA_real_,
                      logHR_D_fixed_SD=NA_real_,error=if(fit$usable) '' else fit$error)
    if(fit$usable) {
      b <- coef(fit$full$fit); row$betaP <- (b['M']-b['D'])/2; row$betaN <- (b['M']+b['D'])/2
      row$betaD <- b['D']; row$logHR_D_fixed_SD <- b['D']*transform$SD_D
    }
    rows[[i]] <- row
  }
  do.call(rbind,rows)
}

verify_analysis_lock <- function(run_root,protocol_sha,d,qualification) {
  path <- file.path(run_root,'statistics','analysis_lock.json'); need(file.exists(path),'Frozen analysis lock required')
  lock <- read_json(path)
  need(identical(lock$protocol_sha256,protocol_sha),'Analysis lock protocol mismatch')
  need(identical(lock$qualification_sha256,qualification$sha256),'Qualification changed after analysis lock')
  need(identical(lock$statistical_code_sha256,sha(SCRIPT_FILE)),'Analysis code changed after lock')
  need(identical(lock$cohort_sha256,sha(lock$cohort_path)) && identical(lock$features_sha256,sha(lock$features_path)),
       'Cohort/measurement changed after analysis lock')
  lock
}

primary_execute <- function(d,p,transform,out) {
  pair <- primary_pair(d)
  need(pair$usable,'Primary nested Cox models not numerically estimable; do not replace with an ordinary penalized LRT')
  result <- pair_summary(pair,transform$SD_D)
  ph <- tryCatch(cox.zph(pair$full$fit, transform='km'),error=function(e) e)
  result$proportional_hazards <- if(inherits(ph,'error')) list(usable=FALSE,error=conditionMessage(ph)) else
      list(usable=TRUE,table=unname(ph$table),row_names=rownames(ph$table),columns=colnames(ph$table))
  result$time_interpretation <- time_mode(d)
  result$adjustment_scope <- 'Age, recorded sex and two measured voxel volumes; not stage/HPV/treatment independent'
  boot <- bootstrap_primary(d,p,transform)
  write.csv(boot,file.path(out,'primary_bootstrap.csv'),row.names=FALSE,na='')
  result$bootstrap <- list(resamples=nrow(boot),usable_fraction=mean(boot$usable),
      percentile_intervals=lapply(boot[,c('betaP','betaN','betaD','logHR_D_fixed_SD'),drop=FALSE],
          function(x) as.list(quantile(x[is.finite(x)],c(.025,.5,.975),names=FALSE))))
  saveRDS(list(reduced=pair$reduced$fit,full=pair$full$fit,processing=transform,protocol_sha256=p$.protocol_sha256),
          file.path(out,'frozen_primary_models.rds'))
  result$model_sha256 <- sha(file.path(out,'frozen_primary_models.rds'))
  write_json(file.path(out,'primary_result.json'),result)
}

supporting_execute <- function(d,p,transform,out,primary_result) {
  results <- list()
  stage <- d[!is.na(d$T) & !is.na(d$N) & nzchar(d$T) & nzchar(d$N),,drop=FALSE]
  stage$Tfactor <- factor(stage$T); stage$Nfactor <- factor(stage$N)
  expected <- 6 + nlevels(stage$Tfactor)-1 + nlevels(stage$Nfactor)-1
  results$nominal_T_N <- if(nlevels(stage$Tfactor) < 2 || nlevels(stage$Nfactor) < 2)
      list(usable=FALSE,reason='Insufficient nominal stage levels') else
      pair_summary(fit_pair(stage,c(primary_terms,'Tfactor','Nfactor'),'D',expected),transform$SD_D)
  d$B <- log(d$VP_cm3+d$VN_cm3); d$R <- log(d$VN_cm3/d$VP_cm3)
  results$volume_allocation <- pair_summary(fit_pair(d,c('AGE10','male','B'),'R',4))
  temporal <- temporal_pair(d,p$models$temporal_change_point_days)
  results$temporal <- pair_summary(temporal)
  results$temporal$formal_claim_permitted <- primary_result$p_value < p$models$alpha && temporal$usable
  results$temporal$interpretation <- if(results$temporal$formal_claim_permitted) 'Prespecified gated secondary' else 'Exploratory prespecified estimates; primary gate not satisfied or temporal model unestimable'
  if(temporal$usable) {
    b <- coef(temporal$full$fit); v <- vcov(temporal$full$fit)
    results$temporal$early_contrast <- linear_effect(b,v,c(D=1),transform$SD_D)
    results$temporal$late_contrast <- linear_effect(b,v,c(D=1,Dlate=1),transform$SD_D)
    results$temporal$change <- linear_effect(b,v,c(Dlate=1),transform$SD_D)
  }
  cut <- p$models$temporal_change_point_days
  results$temporal$risk_information <- list(cut_days=cut,early_deaths=sum(d$OS_EVENT[d$OS_TIME <= cut]),
      late_deaths=sum(d$OS_EVENT[d$OS_TIME > cut]),observed_beyond_cut=sum(d$OS_TIME > cut))
  hpv <- d[!is.na(d$HPV) & d$HPV %in% c('0','1','0.0','1.0'),,drop=FALSE]
  hpv$HPVrecorded <- factor(num(hpv$HPV),levels=c(0,1))
  if(nrow(hpv) && length(unique(hpv$HPVrecorded)) == 2) {
    before <- primary_pair(hpv)
    after <- fit_pair(hpv,c(primary_terms,'HPVrecorded'),'D',7)
    hpv_add <- fit_pair(hpv,c(primary_terms,'D'),'HPVrecorded',7)
    results$HPV_recorded_subset <- list(n=nrow(hpv),deaths=sum(hpv$OS_EVENT),
      before=pair_summary(before,transform$SD_D),after=pair_summary(after,transform$SD_D),
      same_patient_HPV_addition=pair_summary(hpv_add),
      interpretation='Selected literal 0/1 categories, assay meanings unverified; no HPV interaction')
  } else results$HPV_recorded_subset <- list(usable=FALSE,reason='No estimable literal 0/1 subset')
  sensitivities <- list(endpoint_fraction=d$f0P <= .01 & d$f1P <= .01 & d$f0N <= .01 & d$f1N <= .01,
      support_500=d$nP >= 500 & d$nN >= 500,
      exclude_known_original_z_over_3=is.na(d$original_z_spacing_mm) | d$original_z_spacing_mm <= 3)
  for(name in names(sensitivities)) {
    x <- d[!is.na(sensitivities[[name]]) & sensitivities[[name]],,drop=FALSE]
    results[[name]] <- if(nrow(x)) pair_summary(primary_pair(x),transform$SD_D) else list(usable=FALSE,reason='No retained patients')
  }
  write_json(file.path(out,'supporting_results.json'),results)
}

heldout_execute <- function(test,p,models,out,run_root) {
  # No refitting. Model and paired-patient bootstrap use exactly frozen scores.
  target <- file.path(out,'heldout_assessment.json')
  need(!file.exists(target),'Held-out assessment is explicit-once; existing result forbids another invocation')
  marker <- file.path(out,'HELDOUT_STARTED.json')
  write_json(marker,list(at_utc=format(Sys.time(),tz='UTC',usetz=TRUE),records=nrow(test),
      model_sha256=sha(file.path(out,'frozen_primary_models.rds'))))
  result <- list(n=nrow(test),deaths=sum(test$OS_EVENT),censored=sum(1-test$OS_EVENT),
      time_interpretation=time_mode(test),model_refitted=FALSE,thresholds_or_features_retuned=FALSE,
      interpretation='Limited held-out assessment; no independent-test significance requirement')
  if(time_mode(test) == 'authentic_entry_counting_process') {
    result$status <- 'unassessable_entry_aware_discrimination_not_implemented'
    result$reason <- 'Ordinary right-censored concordance is not substituted for delayed-entry validation'
  } else {
    x <- transform_data(test,models$processing)
    x$lp_reduced <- as.numeric(predict(models$reduced,newdata=x,type='lp',reference='zero'))
    x$lp_full <- as.numeric(predict(models$full,newdata=x,type='lp',reference='zero'))
    concord <- function(data) tryCatch(c(
      reduced=concordance(Surv(OS_TIME,OS_EVENT)~lp_reduced,data=data,reverse=TRUE,timewt='n/G2')$concordance,
      full=concordance(Surv(OS_TIME,OS_EVENT)~lp_full,data=data,reverse=TRUE,timewt='n/G2')$concordance),
      error=function(e)c(reduced=NA_real_,full=NA_real_))
    point <- concord(x); set.seed(p$bootstrap$heldout_seed)
    b <- replicate(p$bootstrap$heldout_paired_patient_resamples,concord(x[sample.int(nrow(x),nrow(x),replace=TRUE),,drop=FALSE]))
    valid <- apply(b,2,function(y)all(is.finite(y)))
    result$point <- as.list(point); result$bootstrap_valid_fraction <- mean(valid)
    result$status <- if(all(is.finite(point)) && mean(valid) >= .95) 'limited_assessable' else 'unassessable_or_bootstrap_support_inadequate'
    result$metric <- 'Uno-weighted right-censored concordance; retrospective timing-unverified interpretation'
    if(any(valid)) result$paired_difference_interval <- as.list(quantile(b[2,valid]-b[1,valid],c(.025,.5,.975),names=FALSE))
    write.csv(x[,c('patient_id','OS_TIME','OS_EVENT','lp_reduced','lp_full')],file.path(out,'heldout_frozen_scores.csv'),row.names=FALSE)
  }
  write_json(target,result)
}

fallback_allowed <- function(run_root) {
  need(!file.exists(file.path(run_root,'statistics','PRIMARY_ASSOCIATION_STARTED.json')),
       'Fallback cannot be promoted after the paired real association was inspected')
  need(!file.exists(file.path(run_root,'statistics','primary_result.json')),
       'Fallback cannot be promoted after the paired real result exists')
  q <- file.path(run_root,'qualification','paired_decision.json')
  design <- file.path(run_root,'statistics','design_assessment.json')
  failed <- (file.exists(q) && identical(read_json(q)$passed,FALSE)) ||
            (file.exists(design) && identical(read_json(design)$passed,FALSE))
  need(failed,'Fallback requires documented paired technical/design infeasibility')
}

main <- function() {
  args <- commandArgs(TRUE); need(length(args) %% 2 == 0,'Arguments must be --name value pairs')
  options <- setNames(as.list(args[seq(2,length(args),2)]),sub('^--','',args[seq(1,length(args),2)]))
  need(all(c('protocol','run-root','stage') %in% names(options)), 'Use --protocol PATH --run-root PATH --stage STAGE')
  run <- normalizePath(options[['run-root']]); checked <- verify_protocol(options$protocol,run)
  p <- checked$protocol; p$.protocol_sha256 <- checked$protocol_sha256
  stage <- options$stage; allowed <- c('design','lock','primary','supporting','test','fallback-design','fallback-fit','fallback-test')
  need(stage %in% allowed,'Unknown execution stage')
  fallback <- grepl('^fallback-',stage)
  if(fallback) fallback_allowed(run)
  q <- require_qualification(run,checked$protocol_sha256,fallback,p$paths)
  d <- read_data(p,if(fallback) 'primary_only' else 'paired',if(stage %in% c('test','fallback-test')) 'test' else 'development')
  out <- file.path(run,'statistics'); dir.create(out,showWarnings=FALSE,recursive=TRUE)
  d$analysis_row_id <- d$patient_id
  if(stage == 'test') {
    lock <- verify_analysis_lock(run,checked$protocol_sha256,d,q)
    path <- file.path(out,'frozen_primary_models.rds'); need(file.exists(path),'Frozen models absent')
    result <- read_json(file.path(out,'primary_result.json'))
    need(identical(sha(path),result$model_sha256),'Frozen model identity changed')
    heldout_execute(d,p,readRDS(path),out,run); return(invisible(NULL))
  }
  if(stage != 'fallback-test') {
    trans <- processing(d,fallback); d <- transform_data(d,trans)
  }
  if(stage == 'design') {
    target <- file.path(out,'design_assessment.json'); need(!file.exists(target),'Design already executed; do not repeat bounded simulations')
    write_json(file.path(out,'DESIGN_STARTED.json'),list(protocol_sha256=checked$protocol_sha256,
      data_bindings=attr(d,'input_bindings'),real_phenotype_OS_coefficients_estimated=FALSE,
      simulations_per_scenario=p$simulation$simulations_per_scenario))
    assessment <- design_assessment(d,p,trans)
    write.csv(assessment$replicates,file.path(out,'design_simulations.csv'),row.names=FALSE,na='')
    assessment$replicates <- NULL; assessment$protocol_sha256 <- checked$protocol_sha256
    assessment$processing <- trans; assessment$n <- nrow(d); assessment$deaths <- sum(d$OS_EVENT)
    assessment$time_interpretation <- time_mode(d); assessment$real_phenotype_OS_association_inspected <- FALSE
    assessment$approximation <- p$simulation$baseline_approximation
    calibration <- calibrate_design(d)
    assessment$calibration <- calibration[setdiff(names(calibration),'entry')]
    write_json(target,assessment); return(invisible(NULL))
  }
  if(stage == 'lock') {
    design_path <- file.path(out,'design_assessment.json'); need(file.exists(design_path),'Design assessment absent')
    design <- read_json(design_path); need(isTRUE(design$passed),'Design precision/stability gate failed')
    need(identical(design$protocol_sha256,checked$protocol_sha256),'Design protocol mismatch')
    need(!file.exists(file.path(out,'PRIMARY_ASSOCIATION_STARTED.json')),'Cannot create lock after association inspection')
    write_json(file.path(out,'analysis_lock.json'),list(protocol_sha256=checked$protocol_sha256,
      statistical_code_sha256=sha(SCRIPT_FILE),source_commit=checked$lock$source_commit,
      qualification_sha256=q$sha256,design_sha256=sha(design_path),
      cohort_path=p$paths$cohort_manifest_csv,cohort_sha256=sha(p$paths$cohort_manifest_csv),
      features_path=p$paths$phenotype_features_csv,features_sha256=sha(p$paths$phenotype_features_csv),
      development_IDs=as.list(d$patient_id),processing=trans,models=p$models,
      timing_interpretation=time_mode(d),reserved_test_access='explicit_once_later',
      real_phenotype_OS_association_inspected=FALSE))
    return(invisible(NULL))
  }
  if(stage %in% c('primary','supporting')) {
    lock <- verify_analysis_lock(run,checked$protocol_sha256,d,q)
    need(identical(unlist(lock$development_IDs),d$patient_id),'Development record IDs changed')
    need(identical(lock$design_sha256,sha(file.path(out,'design_assessment.json'))),'Design assessment changed')
    if(stage == 'primary') {
      write_json(file.path(out,'PRIMARY_ASSOCIATION_STARTED.json'),list(protocol_sha256=checked$protocol_sha256,
        analysis_lock_sha256=sha(file.path(out,'analysis_lock.json')),outcome='OS'))
      primary_execute(d,p,trans,out)
    } else {
      result_path <- file.path(out,'primary_result.json'); need(file.exists(result_path),'Primary result absent')
      supporting_execute(d,p,trans,out,read_json(result_path))
    }
    return(invisible(NULL))
  }
  # A separate fallback is intentionally not substituted for the six-slope
  # protocol. Its gate and model are fully explicit and must precede inspection.
  if(stage == 'fallback-design') {
    target <- file.path(out,'fallback_design_assessment.json'); need(!file.exists(target),'Fallback design already executed')
    set.seed(p$simulation$fallback_seed); n <- p$simulation$fallback_simulations_per_scenario
    all_rows <- list(); summaries <- list()
    need(identical(unlist(p$simulation$fallback_scenarios),c('null_phiP','alternative_phiP_HR1_5')),
         'Fallback scenarios must be separately prospectively specified')
    need(mean(d$OS_EVENT)>0 && mean(d$OS_EVENT)<1,'Fallback design requires deaths and censoring')
    calibration <- calibrate_design(d)
    write_json(file.path(out,'FALLBACK_DESIGN_STARTED.json'),list(protocol_sha256=checked$protocol_sha256,
      qualification_sha256=q$sha256,statistical_code_sha256=sha(SCRIPT_FILE),
      data_bindings=attr(d,'input_bindings'),real_phenotype_OS_coefficients_estimated=FALSE,
      simulations_per_scenario=n,scenarios=p$simulation$fallback_scenarios,
      calibration=calibration[setdiff(names(calibration),'entry')]))
    for(scenario in unlist(p$simulation$fallback_scenarios)) {
      rows <- vector('list',n); truth <- if(scenario == 'null_phiP') 0 else log(p$simulation$fallback_alternative_HR_phiP_per_development_SD)
      for(i in seq_len(n)) {
        lp <- log(p$simulation$fallback_nuisance_HR_age_per_10_units)*d$AGE10+
          log(p$simulation$fallback_nuisance_HR_log_primary_volume_per_unit)*(d$logVP-mean(d$logVP))+
          truth*(d$phiP-mean(d$phiP))/trans$SD_phiP
        te <- rexp(nrow(d),calibration$event_rate*exp(lp)); tc <- rexp(nrow(d),calibration$censor_rate)
        x <- d; x$OS_TIME <- pmin(te,tc); x$OS_EVENT <- as.integer(te<=tc)
        # Unknown timing remains exploratory; verified-entry records retain entry.
        x$OS_TIME <- x$OS_TIME+calibration$entry
        pair <- fit_pair(x,c('AGE10','male','logVP'),'phiP',4)
        effect <- if(pair$usable) linear_effect(coef(pair$full$fit),vcov(pair$full$fit),c(phiP=1),trans$SD_phiP) else NULL
        rows[[i]] <- data.frame(scenario=scenario,replicate=i,usable=pair$usable,
          p_value=if(pair$usable)pair$p_value else NA_real_,truth=truth,
          estimate=if(pair$usable)effect$log_HR else NA_real_,
          lower95=if(pair$usable)effect$lower95*trans$SD_phiP else NA_real_,
          upper95=if(pair$usable)effect$upper95*trans$SD_phiP else NA_real_)
      }
      rows <- do.call(rbind,rows); good <- rows[rows$usable,,drop=FALSE]
      coverage <- if(nrow(good))mean(good$lower95<=truth & good$upper95>=truth) else NA_real_
      rejection <- if(nrow(good))mean(good$p_value<p$models$alpha) else NA_real_
      width <- if(nrow(good))median(good$upper95-good$lower95) else NA_real_
      checks <- c(mean(rows$usable)>=p$simulation$minimum_usable_fraction,
        is.finite(coverage)&&coverage>=p$simulation$CI_coverage_range[1]&&coverage<=p$simulation$CI_coverage_range[2],
        is.finite(width)&&width<=p$simulation$maximum_median_log_HR_CI_width,
        scenario!='null_phiP'||(is.finite(rejection)&&rejection<=p$simulation$maximum_null_rejection_fraction))
      summaries[[scenario]] <- list(usable_fraction=mean(rows$usable),coverage=coverage,
        rejection_or_power=rejection,median_interval_width=width,passed=all(checks)); all_rows[[scenario]]<-rows
    }
    write.csv(do.call(rbind,all_rows),file.path(out,'fallback_design_simulations.csv'),row.names=FALSE)
    write_json(target,list(passed=all(vapply(summaries,function(x)x$passed,logical(1))),scenarios=summaries,
      protocol_sha256=checked$protocol_sha256,qualification_sha256=q$sha256,processing=trans,
      input_bindings=attr(d,'input_bindings'),development_IDs=as.list(d$patient_id),
      statistical_code_sha256=sha(SCRIPT_FILE),
      calibration=calibration[setdiff(names(calibration),'entry')],association_inspected=FALSE)); return(invisible(NULL))
  }
  if(stage == 'fallback-fit') {
    path <- file.path(out,'fallback_design_assessment.json'); need(file.exists(path),'Fallback design absent')
    design <- read_json(path); need(isTRUE(design$passed),'Fallback design failed')
    need(identical(design$protocol_sha256,checked$protocol_sha256) &&
         identical(design$qualification_sha256,q$sha256) && identical(design$statistical_code_sha256,sha(SCRIPT_FILE)),
         'Fallback source/protocol/qualification changed after design')
    for(binding in design$input_bindings) if(!is.null(binding))
      need(identical(binding$sha256,sha(binding$path)),'Fallback design inputs changed')
    need(identical(unlist(design$development_IDs),d$patient_id),'Fallback development IDs changed')
    write_json(file.path(out,'fallback_analysis_lock.json'),list(protocol_sha256=checked$protocol_sha256,
      design_sha256=sha(path),qualification_sha256=q$sha256,statistical_code_sha256=sha(SCRIPT_FILE),
      input_bindings=design$input_bindings,development_IDs=design$development_IDs,processing=design$processing,
      real_association_inspected=FALSE))
    write_json(file.path(out,'FALLBACK_ASSOCIATION_STARTED.json'),list(protocol_sha256=checked$protocol_sha256,
      qualification_sha256=q$sha256,design_sha256=sha(path)))
    pair <- fit_pair(d,c('AGE10','male','logVP'),'phiP',4)
    result <- pair_summary(pair); result$time_interpretation <- time_mode(d)
    if(pair$usable) result$sphericity_HR_per_fixed_development_SD <- linear_effect(coef(pair$full$fit),vcov(pair$full$fit),c(phiP=1),trans$SD_phiP)
    result$role <- 'Separate prespecified fallback activated before paired association; not replacement for a null result'
    if(pair$usable) {
      set.seed(p$bootstrap$fallback_association_seed)
      boot <- lapply(seq_len(p$bootstrap$fallback_association_patient_resamples),function(i) {
        x <- resample_patients(d,sample.int(nrow(d),nrow(d),replace=TRUE))
        fit <- fit_pair(x,c('AGE10','male','logVP'),'phiP',4)
        data.frame(replicate=i,usable=fit$usable,coefficient=if(fit$usable)coef(fit$full$fit)['phiP'] else NA_real_,
          logHR_fixed_development_SD=if(fit$usable)coef(fit$full$fit)['phiP']*trans$SD_phiP else NA_real_,
          error=if(fit$usable)'' else fit$error)
      })
      boot <- do.call(rbind,boot); rownames(boot) <- NULL
      write.csv(boot,file.path(out,'fallback_bootstrap.csv'),row.names=FALSE,na='')
      result$bootstrap <- list(resamples=nrow(boot),usable_fraction=mean(boot$usable),
        percentile_logHR_interval=as.list(quantile(boot$logHR_fixed_development_SD[boot$usable],c(.025,.5,.975),names=FALSE)))
      ph <- tryCatch(cox.zph(pair$full$fit,transform='km'),error=function(e)e)
      result$proportional_hazards <- if(inherits(ph,'error'))list(usable=FALSE,error=conditionMessage(ph)) else
        list(usable=TRUE,table=unname(ph$table),row_names=rownames(ph$table),columns=colnames(ph$table))
      saveRDS(list(reduced=pair$reduced$fit,full=pair$full$fit,processing=trans,protocol_sha256=checked$protocol_sha256),
              file.path(out,'frozen_fallback_models.rds'))
      result$model_sha256 <- sha(file.path(out,'frozen_fallback_models.rds'))
    }
    write_json(file.path(out,'fallback_result.json'),result)
  }
  if(stage == 'fallback-test') {
    path <- file.path(out,'fallback_analysis_lock.json'); need(file.exists(path),'Fallback analysis lock absent')
    lock <- read_json(path)
    need(identical(lock$protocol_sha256,checked$protocol_sha256) && identical(lock$qualification_sha256,q$sha256) &&
         identical(lock$statistical_code_sha256,sha(SCRIPT_FILE)),'Fallback frozen identities changed')
    for(binding in lock$input_bindings) if(!is.null(binding))
      need(identical(binding$sha256,sha(binding$path)),'Fallback frozen inputs changed')
    model_path <- file.path(out,'frozen_fallback_models.rds'); result <- read_json(file.path(out,'fallback_result.json'))
    need(file.exists(model_path) && identical(sha(model_path),result$model_sha256),'Fallback model identity changed')
    fallback_out <- file.path(out,'fallback-heldout'); dir.create(fallback_out,showWarnings=FALSE)
    file.copy(model_path,file.path(fallback_out,'frozen_primary_models.rds'),overwrite=FALSE)
    heldout_execute(d,p,readRDS(model_path),fallback_out,run)
  }
}

if(sys.nframe() == 0L) main()
